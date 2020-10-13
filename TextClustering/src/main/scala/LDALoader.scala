import java.io.{File, PrintWriter}
import java.util.Calendar

import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.DistributedLDAModel
import org.apache.spark.mllib.linalg.{SparseVector, Vectors}
import org.apache.spark.sql.SparkSession

import scala.collection.mutable

object LDALoader extends App {
  val lang = "EN";
  val conf = new SparkConf()
    .setAppName(s"LDAClusteringTest")
    .setMaster("local[*]")
    .set("spark.executor.memory", "10g")
    .set("spark.network.timeout", "600s")
    .set("spark.shuffle.memoryFraction", "0.5")
  val spark = SparkSession.builder().config(conf).getOrCreate() //session

  val sc = spark.sparkContext //context
  // Change the log level to warn or info in other to get more information
  sc.setLogLevel("ERROR")
  //if there exists a model in the modelsFolder
  val modelsFolder  = new File("src/main/resources/models")
  val modelFiles = modelsFolder.listFiles
    .filter(_.getName.startsWith("LdaModel_" + lang)) //filter the name structure
    .map(_.getPath).toList //mapping to a list (of one param)

  if (modelFiles != null && modelFiles.length > 0) {
    println("List of models:")
    modelFiles.foreach { model =>
      println(s"Model: $model")
    }

    // Load last trained model
    val LDATrainedModel = DistributedLDAModel.load(sc, modelFiles.last)
    var modelPathParts = modelFiles.last.split('\\')
    if (modelPathParts.length <= 1){
      modelPathParts = modelFiles.last.split('/')
    }

    val globalVocabulary = sc.textFile(s"src/main/resources/models/vocabularies/${modelPathParts.last}").map(_.split(",")).collect()(0)

    // Get test dataset with all books of a language
    var datasetFolderPath = "src/main/resources/books"
    lang match {
      case "DU" => datasetFolderPath = datasetFolderPath + "/Dutch"
      case "EN" => datasetFolderPath = datasetFolderPath + "/English"
      case "FR" => datasetFolderPath = datasetFolderPath + "/French"
      case "GE" => datasetFolderPath = datasetFolderPath + "/German"
      case "IT" => datasetFolderPath = datasetFolderPath + "/Italian"
      case "RU" => datasetFolderPath = datasetFolderPath + "/Russian"
      case "SP" => datasetFolderPath = datasetFolderPath + "/Spanish"
      case "UKR" => datasetFolderPath = datasetFolderPath + "/Ukrainian"
    }

    var textOutputContent = ""
    val stopWordFile: String = s"src/main/resources/stopWords_$lang.txt"
    val stopWordText = sc.textFile(stopWordFile).collect()
    val datasetFolder = new File(datasetFolderPath)
    //mapping to the books to a list
    val books = datasetFolder.listFiles.filter(_.isFile).map(_.getPath).toList

    // Print the topics, showing the top-weighted terms for each topic.
    val topicIndices = LDATrainedModel.describeTopics(maxTermsPerTopic = 300)
    val topics = topicIndices.map { case (terms, termWeights) =>
      terms.zip(termWeights).map { case (term, weight) => (globalVocabulary(term.toInt), weight) }
    }
    textOutputContent = textOutputContent + "#######################################################################################\n"
    textOutputContent = textOutputContent + s"LDA Model with ${LDATrainedModel.k} topics\n"
    textOutputContent = textOutputContent + "#######################################################################################\n"
    println("#######################################################################################")
    println(s"LDA Model with ${LDATrainedModel.k} topics")
    println("#######################################################################################")
    val booksPerTopicCountVector = Array.fill(LDATrainedModel.k){0}
    val booksPerTopicNameVector = Array.fill(LDATrainedModel.k){""}
    //loop on the list of books
    var bookCounter = 0
    books.foreach { book =>
      val bookPath = book.replace(",", "?")
      //vector for testing of a book
      val (documents, localVocab) = TFIDfVectorizer.BuildCountVector(sc, bookPath, LDATrainedModel.vocabSize, stopWordText)

      // Get sorted tokens
      val documentTokens = documents.map{ case (id, vec) =>
        val indices = vec.asInstanceOf[SparseVector].indices
        val tokens = new Array[(String, Double)](localVocab.size)
        for (idLocal <- indices) {
          val tokenAtId = localVocab(idLocal)
          tokens(idLocal) = (tokenAtId, vec(idLocal))
        }
        (tokens)
      }.collect()(0).sortWith((e1, e2) => (e1._2 > e2._2))

      // Calculate TF Matrix
      val tf = documents.map { case (id, vec) =>
        val indices = vec.asInstanceOf[SparseVector].indices
        val counts = new mutable.HashMap[Int, Double]()
        for (idx <- indices) {
          val globalId = globalVocabulary.indexOf(localVocab(idx.toInt))
          counts(globalId) = vec(idx)
        }
        (id, Vectors.sparse(globalVocabulary.size, counts.toSeq))
      }.cache()
      val tfVector = tf.collect()
      //search for topicDistribution based on the vector
      val topicDistributions = LDATrainedModel.toLocal.topicDistribution(tfVector(0)_2)

      //display
      val bookPathParts = bookPath.split('\\')
      textOutputContent = textOutputContent + "***************************************************************************************\n"
      println("***************************************************************************************")
      println(s"Book's number: $bookCounter")
      textOutputContent = textOutputContent + s"Book's number: $bookCounter\n"
      bookCounter = bookCounter + 1
      var bookName = ""
      if (bookPathParts != null && !bookPathParts.isEmpty && bookPathParts.length > 0) {
        bookName = bookPathParts(bookPathParts.length - 1)
        textOutputContent = textOutputContent + s"Book's name: $bookName\n"
        println(s"Book's name: $bookName")
      } else {
        textOutputContent = textOutputContent + s"Book's name: $bookPath\n"
        println(s"Book's name: $bookPath")
      }

      textOutputContent = textOutputContent + "\n-------------------------------------------------------\nTopics Nr. \t|\t Distribution\n-------------------------------------------------------\n"
      println("-------------------------------------------------------")
      println("Topics Nr. \t|\t Distribution")
      println("-------------------------------------------------------")
      var mainTopic = 0
      var mainTopicWeight = 0.0
      topicDistributions.foreachActive((idx, weight) => {
        textOutputContent = textOutputContent + s"Nr.: $idx \t\t|\t ${weight}\n"
        println(s"Nr.: $idx \t\t|\t ${weight}")
        if (mainTopicWeight <= weight) {
          mainTopic = idx
          mainTopicWeight = weight
        }
      })

      booksPerTopicCountVector(mainTopic) = booksPerTopicCountVector(mainTopic) + 1
      booksPerTopicNameVector(mainTopic) = s"${booksPerTopicNameVector(mainTopic)}$bookName"
      if ((booksPerTopicCountVector(mainTopic) % 3) == 0) {
        booksPerTopicNameVector(mainTopic) = s"${booksPerTopicNameVector(mainTopic)}\n"
      }
      else {
        booksPerTopicNameVector(mainTopic) = s"${booksPerTopicNameVector(mainTopic)}, "
      }
      // Show main Topic of the book
      textOutputContent = textOutputContent + s"Main topic of the book: Topic Nr. ($mainTopic), Weight ($mainTopicWeight)\n"
      println(s"Main topic of the book: Topic Nr. ($mainTopic), Weight ($mainTopicWeight)")
      // Print the first 10 words of the book
      val topicVocab = topics(mainTopic).map(_._1)
      val docVocabPart = documentTokens.slice(0, 100).map(_._1).intersect(topicVocab)
      textOutputContent = textOutputContent + "Book most important words\n-------------------------------------------------------\nWord. \t|\t TF\n-------------------------------------------------------\n"
      println("Book most important words")
      println("-------------------------------------------------------")
      println("Most important terms:")
      println("-------------------------------------------------------")
      docVocabPart.slice(0, 10).foreach(w => {
        print(s"${w}, ")
        textOutputContent = textOutputContent + s"${w}, "
      })
      println("\n***************************************************************************************\n")
      textOutputContent = textOutputContent + "\n***************************************************************************************\n\n"

      spark.sqlContext.clearCache()
    }

    println("***************************************************************************************")
    println("List of topics")
    println("***************************************************************************************")
    textOutputContent = textOutputContent + "***************************************************************************************\n"
    textOutputContent = textOutputContent + "List of topics\n"
    textOutputContent = textOutputContent + "***************************************************************************************\n"
    topics.zipWithIndex.foreach { case (topic, i) =>
      textOutputContent = textOutputContent + "-------------------------------------------------------\n"
      textOutputContent = textOutputContent + s"TOPIC $i: top-weighted terms\n"
      textOutputContent = textOutputContent + "-------------------------------------------------------\n"
      println("-------------------------------------------------------")
      println(s"TOPIC $i: top-weighted terms")
      println("-------------------------------------------------------")
      topic.slice(0, 10).foreach { case (term, weight) =>
        textOutputContent = textOutputContent + s"$term\t$weight\n"
        println(s"$term\t$weight")
      }
      textOutputContent = textOutputContent + "\n"
      println()
      println("-------------------------------------------------------")
      println(s"Amount of books in the topic: ${booksPerTopicCountVector(i)}")
      println("-------------------------------------------------------")
      println("List of Books:")
      println("-------------------------------------------------------")
      println(booksPerTopicNameVector(i))
      println("-------------------------------------------------------\n")
      textOutputContent = textOutputContent + "-------------------------------------------------------\n"
      textOutputContent = textOutputContent + s"Amount of books in the topic: ${booksPerTopicCountVector(i)}\n"
      textOutputContent = textOutputContent + "-------------------------------------------------------\n"
      textOutputContent = textOutputContent + "List of Books:\n"
      textOutputContent = textOutputContent + "-------------------------------------------------------\n"
      textOutputContent = textOutputContent + booksPerTopicNameVector(i)
      textOutputContent = textOutputContent + "\n-------------------------------------------------------\n\n"
    }
    println("***************************************************************************************\n")
    textOutputContent = textOutputContent + "***************************************************************************************\n\n"

    println("#######################################################################################")
    textOutputContent = textOutputContent + "#######################################################################################\n"
    val now = Calendar.getInstance()
    val timestamp = now.getTimeInMillis()
    new PrintWriter(s"src/main/resources/TestOutput/Result_$lang" + s"_$timestamp") { write(textOutputContent); close }
  }
  sc.stop()
}