case class Params(
                   input: String = "",  //Path to Trainings set
                   k: Int = 5, //amount of topics
                   maxIterations: Int = 50,
                   docConcentration: Double = -1,
                   topicConcentration: Double = -1,
                   vocabSize: Int = 2900000, //Vocabulary length
                   stopWordText: Array[String] = null,
                   algorithm: String = "em", //used optimizer
                   checkpointDir: Option[String] = None,
                   checkpointInterval: Int = 10)