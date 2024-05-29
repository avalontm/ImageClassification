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
            // Show the image
            var image = Image.Load<Rgba32>(imagePath);

            // Resize the image to the size expected by the model (e.g., 224x224)
            image.Mutate(x => x.Resize(224, 224));

            // Convert the image to byte array
            using (var ms = new MemoryStream())
            {
                image.Save(ms, new JpegEncoder());
                var imageData = new Input
                {
                    Image = ms.ToArray(),
                    ImagePath = imagePath
                };

                // Check if the model is loaded and the prediction engine is initialized
                if (_predictionEngine != null)
                {
                    var prediction = _predictionEngine.Predict(imageData);

                    // Check the maximum score among classes
                    var maxScore = prediction.Score.Max();

                    // Check if the maximum score is above the confidence threshold
                    if (maxScore >= confidenceThreshold)
                    {
                        prediction.Probability = maxScore;
                        return prediction;
                    }
                    else
                    {
                        // If the maximum score is below the confidence threshold, return a default output
                        return new Output { PredictedLabel = "Imagen no identificada con suficiente confianza", Probability = 0.0f };
                    }
                }
                else
                {
                    // If the prediction engine is not initialized, return a default output
                    return new Output { PredictedLabel = "Modelo no cargado", Probability = 0.0f };
                }
            }
        }
    }
}
