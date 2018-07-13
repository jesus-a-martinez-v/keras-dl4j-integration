import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.nd4j.linalg.io.ClassPathResource

object Dl4JModel extends App {
  // Path to relevant files.
  val jsonModelPath = "/Users/jesusmartinez/Portfolio/keras-dl4j-integration/src/main/iris_model_json"
  val weightsPath = "/Users/jesusmartinez/Portfolio/keras-dl4j-integration/src/main/iris_model_save"

  // Imported model
  val model = KerasModelImport.importKerasSequentialModelAndWeights(jsonModelPath, weightsPath)

  // Define how we should interpret the data.
  val numberOfLinesToSkip = 0
  val delimiter = ','
  val recordReader = new CSVRecordReader(numberOfLinesToSkip, delimiter)
  recordReader.initialize(new FileSplit(new ClassPathResource("iris-encoded.data").getFile))

  // Load the data in memory as it is small.
  val labelIndex = 4
  val numberOfClasses = 3
  val batchSize = 150

  val dataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numberOfClasses)

  val data = dataSetIterator.next()

  data.shuffle()

  // No training required. It was done by Keras.
  // Let's evaluate the model.
  val evaluator = new Evaluation(numberOfClasses)
  val outout = model.output(data.getFeatureMatrix)

  evaluator.eval(data.getLabels, outout)

  System.out.println(evaluator.stats())
}
