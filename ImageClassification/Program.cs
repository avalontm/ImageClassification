﻿namespace ImageClassification
{
    internal class Program
    {
        static string imagesFolderPath = "testing";
        static string modelPath = "model_numbers.zip";
        static string? imagePath;

        static void Main(string[] args)
        {
            MenuBuild();
        }

        static void TitleBuild()
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"Ejercicio de machine learning - Clasificaion de imagenes");
            Console.ResetColor();
        }

        static void MenuBuild()
        {
            Console.Clear();
            TitleBuild();
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"Seleciona una opcion:");
            Console.ResetColor();
            Console.WriteLine($"F1) Entrenar");
            Console.WriteLine($"F2) Probar");
            Console.WriteLine($"ESC) Salir");
            Console.WriteLine();

            ConsoleKey key = Console.ReadKey(true).Key;

            while (key != ConsoleKey.Escape)
            {
                switch (key)
                {
                    case ConsoleKey.F1:
                        onEntrenar();
                        break;
                    case ConsoleKey.F2:
                        onPredecir();
                        break;
                    default:
                        Console.WriteLine($"Opcion no valida.");
                        break;
                }
                MenuBuild();
            }

        }

        static void onEntrenar()
        {
            var modelBuilder = new ModelBuilder(imagesFolderPath, modelPath);
            modelBuilder.BuildAndTrainModel();
            Console.WriteLine();
            Console.WriteLine("Entrenamiento del modelo completo, Preciona cualquier tecla para continuar...");
            Console.ReadKey();
        }

        static void onPredecir()
        {
            Console.Write($"Escribe la ruta de la imagen: ");
            imagePath = Console.ReadLine();

            if (string.IsNullOrEmpty(imagePath) || !File.Exists(imagePath))
            {
                onPredecir();
                return;
            }

            ImagePredictor predictor = new ImagePredictor(modelPath);
            var result = predictor.Predict(imagePath, 0.75f);
            Console.WriteLine();
            Console.WriteLine($"Prediccion: {result.PredictedLabel} | Probabilidad: {result.Probability}");
            Console.WriteLine($"Preciona cualquier tecla para continuar...");
            Console.ReadKey();
        }
    }
}
