using Microsoft.ML;
using Microsoft.ML.Vision;

namespace ImageClassification
{
    public class Input
    {
        public byte[]? Image;
        public string? ImagePath;
        public string? Label;
    }

    public class Output
    {
        public float[]? Score;
        public string? PredictedLabel;
        public float Probability;
    }

    public class ModelBuilder
    {
        private readonly string? _imagePath;
        private readonly string? _modelPath;
        private readonly MLContext _mlContext;
        private readonly string _predictedLabelColumnName = "PredictedLabel";
        private readonly string _keyColumnName = "LabelAsKey";
        IDataView? trainData;

        public ModelBuilder(string imagesFolderPath, string modelPath)
        {
            _imagePath = imagesFolderPath;
            _modelPath = modelPath;
            _mlContext = new MLContext();
        }

        public void BuildAndTrainModel()
        {
            // Creamos un DataView que contenga las rutas y etiquetas de las imágenes.
            var input = LoadLabeledImagesFromPath(_imagePath);
            var data = _mlContext.Data.LoadFromEnumerable(input);
            data = _mlContext.Data.ShuffleRows(data);

            // Cargamos las imágenes y convierta las etiquetas en claves para que sirvan como valores categóricos.
            var images = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(Input.Label), outputColumnName: _keyColumnName)
                .Append(_mlContext.Transforms.LoadRawImageBytes(inputColumnName: nameof(Input.ImagePath), outputColumnName: nameof(Input.Image), imageFolder: _imagePath))
                .Fit(data).Transform(data);

            // Dividimos el conjunto de datos para entrenamiento y prueba.
            var trainTestData = _mlContext.Data.TrainTestSplit(images, testFraction: 0.2, seed: 1);
            trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // Creamos un canal de clasificación de imágenes y entrene el modelo.
            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = nameof(Input.Image),
                LabelColumnName = _keyColumnName,
                ValidationSet = testData,
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101, //DNN previamente entrenado
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                TestOnTrainSet = false
            };

            var pipeline = _mlContext.MulticlassClassification.Trainers.ImageClassification(options)
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue(_predictedLabelColumnName));

            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainData);

            // Evaluamos el modelo y mostrar los resultados.
            var predictions = model.Transform(testData);
            var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: _keyColumnName, predictedLabelColumnName: _predictedLabelColumnName);

            Console.WriteLine();
            Console.WriteLine($"Macro accuracy = {metrics.MacroAccuracy:P2}");
            Console.WriteLine($"Micro accuracy = {metrics.MicroAccuracy:P2}");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine();

            SaveModel(model);
        }

        private static List<Input> LoadLabeledImagesFromPath(string path)
        {
            var images = new List<Input>();
            var directories = Directory.EnumerateDirectories(path);

            foreach (var directory in directories)
            {
                var files = Directory.EnumerateFiles(directory);

                images.AddRange(files.Select(x => new Input
                {
                    ImagePath = Path.GetFullPath(x),
                    Label = Path.GetFileName(directory)
                }));
            }

            return images;
        }


        private void SaveModel(ITransformer model)
        {
            // Guardamos el modelo
            Console.WriteLine();
            Console.WriteLine("Guardando el modelo...");
            _mlContext.Model.Save(model, trainData.Schema, _modelPath);
        }
    }
}
