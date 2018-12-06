using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Transforms.Conversions;
using System;
using System.IO;

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
            Console.WriteLine("Would you like to load the new model (1) or run the existing model (0)?");
            bool loadModel = Convert.ToBoolean(Console.ReadLine());
            if (loadModel)
            {
                // STEP 2: Create a ML.NET environment  
                var mlContext = new MLContext();

                // If working in Visual Studio, make sure the 'Copy to Output Directory'
                // property of iris-data.txt is set to 'Copy always'
                string dataPath = "iris-data.txt";
                var reader = mlContext.Data.TextReader(new TextLoader.Arguments()
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
                    .Append(mlContext.Ranking.Trainers.FastTree(labelColumn: "Label", featureColumn: "Features"))
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
                // STEP 4: Train your model based on the data set  
                var model = pipeline.Fit(trainingDataView);

                using (var stream = File.Create("D:\\code\\LogisticBigModel_TN_FastTree.zip"))
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
            using (var stream = File.OpenRead("D:\\code\\LogisticBigModel_TN_FastTree.zip"))
                model = mlContext.Model.Load(stream);
            while (1 == 1)
            {

                Predict(mlContext, model);
            }
        }

        public static void Predict(MLContext mlContext, ITransformer model)
        {
            Console.WriteLine("What rating does this guy have?");
            int rating = Convert.ToInt32(Console.ReadLine());
            var predictionFunction = model.MakePredictionFunction<RecruitData, RecruitLabel>(mlContext); 
            var prediction =predictionFunction.Predict(
                new RecruitData()
                {
                    Lat = (float)36.1495,
                    Lng = (float)86.4003,
                    Rating = rating,
                });
            Console.WriteLine($"Predicted school  is: {prediction.RecruitPrediction}");
        }
    }
}