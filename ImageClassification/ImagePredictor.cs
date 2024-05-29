using Microsoft.ML.Data;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Formats.Jpeg;
using NumSharp.Utilities;

namespace ImageClassification
{

    public class ImagePredictor
    {
        private readonly string _modelPath;
        private readonly MLContext _mlContext;
        private ITransformer? _model;
        private PredictionEngine<Input,Output>? _predictionEngine;

        public ImagePredictor(string modelPath)
        {
            _modelPath = modelPath;
            _mlContext = new MLContext();
            LoadModel();
        }

        private void LoadModel()
        {
            DataViewSchema modelSchema;
            _model = _mlContext.Model.Load(_modelPath, out modelSchema);
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<Input, Output>(_model);
        }

        public Output Predict(string imagePath, float confidenceThreshold)
        {
            // Cargamos la imagen
            var image = Image.Load<Rgba32>(imagePath);

            // CAmbiamos el tamaño de la imagen al tamaño esperado por el modelo (por ejemplo, 224x224)
            image.Mutate(x => x.Resize(224, 224));

            // Convertir la imagen a una matriz de bytes
            using (var ms = new MemoryStream())
            {
                image.Save(ms, new JpegEncoder());
                var imageData = new Input
                {
                    Image = ms.ToArray(),
                    ImagePath = imagePath
                };

                // Comprobamos si el modelo está cargado y el motor de predicción está inicializado.
                if (_predictionEngine != null)
                {
                    var prediction = _predictionEngine.Predict(imageData);

                    // Consultamos la puntuación máxima entre clases.
                    var maxScore = prediction.Score.Max();

                    // Comprobamos si la puntuación máxima está por encima del umbral de confianza
                    if (maxScore >= confidenceThreshold)
                    {
                        prediction.Probability = maxScore;
                        return prediction;
                    }
                    else
                    {
                        // Si la puntuación máxima está por debajo del umbral de confianza, devolverá un resultado predeterminado
                        return new Output { PredictedLabel = "Imagen no identificada con suficiente confianza", Probability = 0.0f };
                    }
                }
                else
                {
                    // Si el motor de predicción no está inicializado, devuelve una salida predeterminada
                    return new Output { PredictedLabel = "Modelo no cargado", Probability = 0.0f };
                }
            }
        }
    }
}
