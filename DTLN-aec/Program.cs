using DTLN_aec;

//if (args.Length < 6)
//{
//    Console.WriteLine("Usage: DTLNAECProcessor -i <input_folder> -o <output_folder> -m <model_path>");
//    Console.WriteLine("Example: DTLNAECProcessor -i ./input -o ./output -m ./model");
//    return;
//}

string inputFolder = "input";
string outputFolder = "ouput";
string modelPath = "dtln_aec_256";

// Parse command line arguments
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "-i":
        case "--in_folder":
            if (i + 1 < args.Length)
                inputFolder = args[++i];
            break;
        case "-o":
        case "--out_folder":
            if (i + 1 < args.Length)
                outputFolder = args[++i];
            break;
        case "-m":
        case "--model":
            if (i + 1 < args.Length)
                modelPath = args[++i];
            break;
    }
}

// Validate arguments
if (string.IsNullOrEmpty(inputFolder) || string.IsNullOrEmpty(outputFolder) || string.IsNullOrEmpty(modelPath))
{
    Console.WriteLine("Error: All parameters (input folder, output folder, and model path) are required.");
    return;
}

if (!Directory.Exists(inputFolder))
{
    Console.WriteLine($"Error: Input folder '{inputFolder}' does not exist.");
    return;
}

if (!File.Exists($"{modelPath}_1.tflite") || !File.Exists($"{modelPath}_2.tflite"))
{
    Console.WriteLine($"Error: Model files '{modelPath}_1.tflite' and '{modelPath}_2.tflite' not found.");
    return;
}

try
{
    // Set environment variable to make GPUs invisible (equivalent to Python version)
    Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "");

    using var processor = new AudioProcessor(modelPath);
    processor.ProcessFolder(inputFolder, outputFolder);
    Console.WriteLine("Processing completed successfully!");
}
catch (Exception ex)
{
    Console.WriteLine($"Error during processing: {ex.Message}");
    Console.WriteLine($"Stack trace: {ex.StackTrace}");
}
