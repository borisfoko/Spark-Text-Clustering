import java.io.PrintWriter
import java.util.Calendar

import edu.stanford.nlp.process.Morphology
import edu.stanford.nlp.simple.Document
import opennlp.tools.stemmer.PorterStemmer
import opennlp.tools.tokenize.SimpleTokenizer
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.{DistributedLDAModel, EMLDAOptimizer, LDA, OnlineLDAOptimizer}
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.JavaConverters._
import scala.collection.mutable

class LDAClustering(sc: SparkContext, spark: SparkSession, lang: String) {

  def run(params: Params): Unit = {
    // Load documents, and prepare them for LDA.
    val preprocessStart = System.nanoTime()
    val (corpus, vocabArray, actualNumTokens) = TFIDfVectorizer.BuildTFIDFVector(sc, params.input, params.vocabSize, params.stopWordText)
    val actualCorpusSize = corpus.count()
    val actualVocabSize = vocabArray.length
    val preprocessElapsed = (System.nanoTime() - preprocessStart) / 1e9
    corpus.cache()
    println()
    println(s"Corpus summary:")
    println(s"\t Training set size: $actualCorpusSize documents")
    println(s"\t Vocabulary size: $actualVocabSize terms")
    println(s"\t Training set size: $actualNumTokens tokens")
    println(s"\t Preprocessing time: $preprocessElapsed sec")
    println()

    // Define LDA.
    val lda = new LDA()

    // Select optimizer
    val optimizer = params.algorithm.toLowerCase match {
      case "em" => new EMLDAOptimizer
      // add (1.0 / actualCorpusSize) to MiniBatchFraction be more robust on tiny datasets.
      case "online" => new OnlineLDAOptimizer().setMiniBatchFraction(0.05 + 1.0 / actualCorpusSize)
      case _ => throw new IllegalArgumentException(
        s"Only em, online are supported but got ${params.algorithm}.")
    }

    // Set up lda config
    lda.setOptimizer(optimizer)
      .setK(params.k)
      .setMaxIterations(params.maxIterations)
      .setDocConcentration(params.docConcentration)
      .setTopicConcentration(params.topicConcentration)
      .setCheckpointInterval(params.checkpointInterval)
    if (params.checkpointDir.nonEmpty) {
      sc.setCheckpointDir(params.checkpointDir.get)
    }
    val startTime = System.nanoTime()
    // run LDA model training
    println("LDA model training started")
    val ldaModel = lda.run(corpus)
    val elapsed = (System.nanoTime() - startTime) / 1e9
    println(s"Finished training LDA model.  Summary:")
    println(s"\t Training time: $elapsed sec")
    corpus.unpersist()

    val now = Calendar.getInstance()
    val timestamp = now.getTimeInMillis()
    // Save LDA Model
    ldaModel.save(sc, "src/main/resources/models/LdaModel_" + lang + "_" + timestamp)
    new PrintWriter(s"src/main/resources/models/vocabularies/LdaModel_$lang" + s"_$timestamp")
    { write(vocabArray.mkString(",")); close }
    if (ldaModel.isInstanceOf[DistributedLDAModel]) {
      val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
      val avgLogLikelihood = distLDAModel.logLikelihood / actualCorpusSize.toDouble
      println(s"\t Training data average log likelihood: $avgLogLikelihood")
      println()
    }

    // Print the topics, showing the top-weighted terms for each topic.
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    val topics = topicIndices.map { case (terms, termWeights) =>
      terms.zip(termWeights).map { case (term, weight) => (vocabArray(term.toInt), weight) }
    }
    println(s"${params.k} topics:")
    topics.zipWithIndex.foreach { case (topic, i) =>
      println(s"TOPIC $i")
      topic.foreach { case (term, weight) =>
        println(s"$term\t$weight")
      }
      println()
    }

    // Stop the spark context
    sc.stop()
  }
}

object TFIDfVectorizer {
  /**
   * Load documents, tokenize them, create vocabulary, and calculate the TF*IDF matrix.
   *
   * @return (corpus, vocabulary as array, total token count in corpus)
   */
  def BuildTFIDFVector(
                      sc: SparkContext,
                      paths: String,
                      vocabSize: Int,
                      stopWordText: Array[String]
                      ): (RDD[(Long, Vector)], Array[String], Long) = {

    //Reading the Whole Text Files
    val initialRDD = sc.wholeTextFiles(paths).map(_._2)
    //spark forcing to save the data on the hard disk !!
    initialRDD.cache()
    val rdd = initialRDD.mapPartitions { partition =>
      val morphology = new Morphology()
      partition.map { value =>
        LDAUtil.getLemmaText(value, morphology)
      }
    }.map(LDAUtil.filterSpecialCharacters)
    rdd.cache()
    initialRDD.unpersist()

    val stopWordsList: Array[String] = if (stopWordText == null || stopWordText.isEmpty) {
      Array.empty[String]
    } else {
      stopWordText.flatMap(_.stripMargin.split(","))
    }

    // Tokenize RDD
    val tokenized: RDD[(Long, Array[String])] = rdd.zipWithIndex().map { case (text,id) =>
      val tokenizer = SimpleTokenizer.INSTANCE
      val stemmer = new PorterStemmer()
      val tokens = tokenizer.tokenize(text)
      val words = tokens.filter(w => (w.length >= 1) && (!stopWordsList.contains(w)))
        .map(w => stemmer.stem(w))
      id -> words
    }.filter(_._2.length > 0)

    tokenized.cache()

    // Get wordCounts of documents
    val wordCounts: RDD[(String, Long)] = tokenized.flatMap { case (_, tokens) =>
      tokens.map(_ -> 1L)
    }.reduceByKey(_ + _)
    wordCounts.cache()
    val (vocab: Map[String, Int], selectedTokenCount: Long) = {
      val sortedWC: Array[(String,Long)] = {wordCounts.sortBy(_._2, ascending=false).take(vocabSize)}
      (sortedWC.map(_._1).zipWithIndex.toMap, sortedWC.map(_._2).sum)
    }

    // Get tokenized documents
    val documents = tokenized.map { case (id, tokens) =>
      // Filter tokens by vocabulary, and create word count vector representation of document.
      val wc = new mutable.HashMap[Int, Int]()
      tokens.foreach { term =>
        if (vocab.contains(term)) {
          val termIndex = vocab(term)
          wc(termIndex) = wc.getOrElse(termIndex, 0) + 1
        }
      }
      val indices = wc.keys.toArray.sorted
      val values = indices.map(i => wc(i).toDouble)
      val sb = Vectors.sparse(vocab.size, indices, values)
      (id, sb)
    }

    // Get Vocabulary array
    val vocabArray = new Array[String](vocab.size)
    vocab.foreach { case (term, i) => vocabArray(i) = term }

    // Calculate TF Matrix
    val tf = documents.map { case (id, vec) => vec }.cache()

    // Calculate IDf Matrix
    val idfVals = new IDF(2).fit(tf).idf.toArray

    // Calculate TF*IDF Matrix
    val tfidfDocs: RDD[(Long, Vector)] = documents.map { case (id, vec) =>
      val indices = vec.asInstanceOf[SparseVector].indices
      val counts = new mutable.HashMap[Int, Double]()
      for (idx <- indices) {
        var idfValue = idfVals(idx)
        if (idfValue == 0.0) {
          idfValue = 0.0001
        }
        counts(idx) = vec(idx) * idfValue
        //println("TF: " + vec(idx) + ", IDF: " + idfValue + ", TF*IDF: " + counts(idx))
      }
      (id, Vectors.sparse(vocab.size, counts.toSeq))
    }

    // return TF*IDF Matrix of the documents, the vocabulary (token list) and total token Count
    (tfidfDocs, // TF*IDF Matrix
      vocabArray, // vocabulary
      tfidfDocs.map(_._2.numActives).sum().toLong) // total token count
  }

  /**
   * Load documents, tokenize them, create vocabulary, and countVector.
   *
   * @return (corpus, vocabulary as array)
   */
  def BuildCountVector(
                        sc: SparkContext,
                        paths: String,
                        vocabSize: Int,
                        stopWordText: Array[String]
                      ): (RDD[(Long, Vector)], Array[String]) = {

    //Reading the Whole Text Files
    val initialRDD = sc.wholeTextFiles(paths).map(_._2)
    //spark forcing to save the data on the hard disk !!
    initialRDD.cache()
    val rdd = initialRDD.mapPartitions { partition =>
      val morphology = new Morphology()
      partition.map { value =>
        LDAUtil.getLemmaText(value, morphology)
      }
    }.map(LDAUtil.filterSpecialCharacters)
    rdd.cache()
    initialRDD.unpersist()

    val stopWordsList: Array[String] = if (stopWordText == null || stopWordText.isEmpty) {
      Array.empty[String]
    } else {
      stopWordText.flatMap(_.stripMargin.split(","))
    }

    val tokenized: RDD[(Long, Array[String])] = rdd.zipWithIndex().map { case (text,id) =>
      val tokenizer = SimpleTokenizer.INSTANCE
      val stemmer = new PorterStemmer()
      val tokens = tokenizer.tokenize(text)
      val words = tokens.filter(w => (w.length >= 1) && (!stopWordsList.contains(w)))
        .map(w => stemmer.stem(w))
      id -> words
    }.filter(_._2.length > 0)

    tokenized.cache()

    // Get RDD of documents
    val wordCounts: RDD[(String, Long)] = tokenized.flatMap { case (_, tokens) =>
      tokens.map(_ -> 1L)
    }.reduceByKey(_ + _)
    wordCounts.cache()
    val (vocab: Map[String, Int], selectedTokenCount: Long) = {
      val sortedWC: Array[(String,Long)] = {wordCounts.sortBy(_._2, ascending=false).take(vocabSize)}
      (sortedWC.map(_._1).zipWithIndex.toMap, sortedWC.map(_._2).sum)
    }

    // Get tokenized documents
    val documents = tokenized.map { case (id, tokens) =>
      // Filter tokens by vocabulary, and create word count vector representation of document.
      val wc = new mutable.HashMap[Int, Int]()
      tokens.foreach { term =>
        if (vocab.contains(term)) {
          val termIndex = vocab(term)
          wc(termIndex) = wc.getOrElse(termIndex, 0) + 1
        }
      }
      val indices = wc.keys.toArray.sorted
      val values = indices.map(i => wc(i).toDouble)
      val sb = Vectors.sparse(vocab.size, indices, values)
      (id, sb)
    }

    // Get Vocabulary array
    val vocabArray = new Array[String](vocab.size)
    vocab.foreach { case (term, i) => vocabArray(i) = term }

    // return Word list and Word Count
    (documents, // RDD Vectors
      vocabArray) // vocabulary
  }

}

//cleaning of the texts and lemma of the words
object LDAUtil {

  //to split the texts in the books - replace of the special signs
  def filterSpecialCharacters(document: String) = document.replaceAll(
  """[» « ! @ # $ % ^ & * ( ) _ + - − , ” " ’ ' ; : . ` ? --]""", " ")

  //Stamm der Wörter - nicht genutzt
  def getStemmedText(document: String, lang: String) = {
    val morphology = new Morphology()
    new Document(document).sentences().asScala.toList.flatMap(_.words().asScala.toList.map(morphology.stem)).mkString(" ")
  }

  //konjugierte Wörter werden zusammen gebracht und als ein Wort erkannt (nur englisch)
  def getLemmaText(document: String, morphology: Morphology) = {
    val string = new StringBuilder()
    val value = new Document(document).sentences().asScala.toList.flatMap { a =>
      val words = a.words().asScala.toList
      val tags = a.posTags().asScala.toList
      (words zip tags).toMap.map { a =>
        val newWord = morphology.lemma(a._1, a._2)
        val addedWord = if (newWord.length > 3) {
          newWord
        } else {
          ""
        }
        string.append(addedWord + " ")
      }
    }
    string.toString()
  }
}