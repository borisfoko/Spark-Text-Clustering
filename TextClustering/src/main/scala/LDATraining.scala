import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

//create object LDATraining
object LDATraining extends App {
    val lang = "EN"
    val conf = new SparkConf().setAppName(s"LDATraining").setMaster("local[*]")
      .set("spark.executor.memory", "12g")
      .set("spark.network.timeout", "700s")
      .set("spark.shuffle.memoryFraction", "0.6")
    val spark = SparkSession.builder().config(conf).getOrCreate()

    val sc = spark.sparkContext
    // Change the log level to warn or info in other to get more information
    sc.setLogLevel("ERROR")
    val lda = new LDAClustering(sc, spark, lang)
    //stopWordFile: String = "",
    // Exclude function's words (don't assign them to any topic)
    val stopWordFile: String = s"src/main/resources/stopWords_$lang.txt"
    val defaultParams = Params().copy(input = "src/main/resources/books/English", stopWordText = sc.textFile(stopWordFile).collect())
    lda.run(defaultParams)
}