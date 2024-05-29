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

        public Output Predict(string imagePath)
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
                var prediction = _predictionEngine.Predict(imageData);

                return prediction;
            }
        }
    }
}
