using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq; 
namespace myApp
{
    class Program
    {
        // STEP 1: Define your data structures
        // IrisData is used to provide training data, and as
        // input for prediction operations
        // - First 4 properties are inputs/features used to predict the label
        // - Label is what you are predicting, and is only set when training
        public class RecruitData
        {
            public float Lat;
            public float Lng;
            public float Rating;
            public string Label;
        }

        // IrisPrediction is the result returned from prediction operations
        public class RecruitLabel
        {
            [ColumnName("PredictedLabel")]
            public string RecruitPrediction;
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Would you like to load the new model (true) or run the existing model (false)?");
            bool loadModel = true; 
            bool.TryParse(Console.ReadLine(), out loadModel); 
            if (loadModel)
            {
                // STEP 2: Create a ML.NET environment  
                var mlContext = new MLContext();

                // If working in Visual Studio, make sure the 'Copy to Output Directory'
                // property of iris-data.txt is set to 'Copy always'
                string dataPath = "iris-data.txt";
                var reader = mlContext.Data.CreateTextReader(new TextLoader.Arguments()
                {
                    Separator = ",",
                    HasHeader = false,
                    Column = new[]
                    {
                        new TextLoader.Column("Lat", DataKind.R4, 0),
                        new TextLoader.Column("Lng", DataKind.R4, 1),
                        new TextLoader.Column("Rating", DataKind.R4, 2),
                        new TextLoader.Column("Label", DataKind.Text, 3),
                    }
                });


                IDataView trainingDataView = reader.Read(new MultiFileSource(dataPath));

                // STEP 3: Transform your data and add a learner
                // Assign numeric values to text in the "Label" column, because only
                // numbers can be processed during model training.
                // Add a learning algorithm to the pipeline. e.g.(What type of iris is this?)
                // Convert the Label back into original text (after converting to number in step 3)

                var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                    .Append(mlContext.Transforms.Concatenate("Features", "Lat", "Lng", "Rating"))
                    .Append(mlContext.MulticlassClassification.Trainers.LogisticRegression(labelColumn: "Label", featureColumn: "Features"))
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
                // STEP 4: Train your model based on the data set  
                var (trainData, testData) = mlContext.MulticlassClassification.TrainTestSplit(trainingDataView, testFraction: 0.2);
                var model = pipeline.Fit(trainingDataView);
                var metrics = mlContext.MulticlassClassification.Evaluate(model.Transform(testData));
                Console.WriteLine("Accuracy: " + metrics.AccuracyMicro);
                //var cvResults = mlContext.MulticlassClassification.CrossValidate(trainingDataView, pipeline, numFolds: 5);
                //var microAccuracies = cvResults.Select(r => r.metrics.AccuracyMicro);
                //Console.WriteLine(microAccuracies.Average());
                using (var stream = File.Create(Directory.GetCurrentDirectory()+"/LogisticBigModel_TN_FastTree.zip"))
                {
                    // Saving and loading happens to 'dynamic' models.
                    mlContext.Model.Save(model, stream);
                }
            }
            // STEP 5: Use your model to make a prediction
            // You can change these numbers to test different predictions
            LoadAndRunModel();

        }

        private static void LoadAndRunModel()
        {
            var mlContext = new MLContext();
            ITransformer model;
            using (var stream = File.OpenRead(Directory.GetCurrentDirectory() + "/LogisticBigModel_TN_FastTree.zip"))
                model = mlContext.Model.Load(stream);
            while (1 == 1)
            {

                Predict(mlContext, model);
            }
        }

        private static Tuple <float, float> GetCoordinatesForCity()
        {
            Console.WriteLine("1. Knoxville 2.) Chattanogga 3.) Memphis 4.) Nashville"); 

            string city = Console.ReadLine(); 
             
            if (city == "1")
                return new Tuple<float, float>((float)35.9606, (float)83.9207);
            if (city == "2")
                return new Tuple<float, float>((float)35.0456, (float)85.3097);
            if (city == "3")
                return new Tuple<float, float>((float)35.1495, (float)90.0490);
            if (city == "4")
                return new Tuple<float, float>((float)36.1627, (float)86.7816);
            else
                return new Tuple<float, float>((float)36.1628, (float)85.5016); 
        }

        public static void Predict(MLContext mlContext, ITransformer model)
        {
            Console.WriteLine("What rating does this recruit have?");
            int rating = 99; 
            int.TryParse(Console.ReadLine(), out rating); 

            Console.WriteLine("What city is this recruit from?");
            Tuple<float, float> coordinates = GetCoordinatesForCity();

            
            var predictionFunction = model.CreatePredictionEngine<RecruitData, RecruitLabel>(mlContext);
            var prediction = predictionFunction.Predict(
                new RecruitData()
                {
                    Lat = coordinates.Item1,
                    Lng = coordinates.Item2,
                    Rating = rating,
                });
            Console.WriteLine($"Predicted school  is: {prediction.RecruitPrediction}");
        }
    }
}