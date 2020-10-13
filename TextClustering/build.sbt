name := "TextClustering"

version := "1.0"

scalaVersion := "2.12.10"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.12" % "2.4.3",
  "org.apache.spark" % "spark-sql_2.12" % "2.4.3",
  "org.apache.spark" % "spark-mllib_2.12" % "2.4.3",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.6.0",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.6.0" classifier "models",
  "com.google.protobuf" % "protobuf-java" % "2.6.1",
  "org.apache.opennlp" % "opennlp-tools" % "1.6.0"
)