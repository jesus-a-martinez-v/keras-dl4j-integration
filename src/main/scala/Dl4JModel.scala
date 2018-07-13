import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.nd4j.linalg.io.ClassPathResource

object Dl4JModel extends App {
  val jsonModelPath = "/Users/jesusmartinez/Portfolio/keras-dl4j-integration/src/main/iris_model_json"
  val weightsPath = "/Users/jesusmartinez/Portfolio/keras-dl4j-integration/src/main/iris_model_save"

  val model = KerasModelImport.importKerasSequentialModelAndWeights(jsonModelPath, weightsPath)

  val numberOfLinesToSkip = 0
  val delimiter = ','
  val recordReader = new CSVRecordReader(numberOfLinesToSkip, delimiter)
  recordReader.initialize(new FileSplit(new ClassPathResource("iris-encoded.data").getFile))

  val labelIndex = 4
  val numClasses = 3
  val batchSize = 150

  val dataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses)

  val data = dataSetIterator.next()

  data.shuffle()

  val eval = new Evaluation(3)
  val outout = model.output(data.getFeatureMatrix)
  eval.eval(data.getLabels, outout)
  System.out.println(eval.stats())
}
