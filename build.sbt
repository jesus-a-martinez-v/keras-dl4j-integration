name := "keras-dl4j-integration"

version := "0.1"

scalaVersion := "2.12.4"

classpathTypes += "maven-plugin"

libraryDependencies ++= Seq(
  "org.nd4j" % "nd4j-native-platform" % "0.9.1",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.9.1",
  "com.typesafe" % "config" % "1.3.3",
  "org.deeplearning4j" % "deeplearning4j-modelimport" % "0.9.1"
)