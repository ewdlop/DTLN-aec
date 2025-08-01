using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;
using Numpy;

namespace DTLN_aec
{
    public class AudioProcessor : IDisposable
    {
        private InferenceSession _session1;
        private InferenceSession _session2;
        private const int BlockLength = 512;
        private const int BlockShift = 128;
        private const int SampleRate = 16000;

        public AudioProcessor(string modelPath)
        {
            try
            {
                // Load the two ONNX model parts
                var sessionOptions = new SessionOptions();
                _session1 = new InferenceSession($"{modelPath}_1.onnx", sessionOptions);
                _session2 = new InferenceSession($"{modelPath}_2.onnx", sessionOptions);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading models: {ex.Message}");
                throw;
            }
        }

        public void ProcessFile(string audioFileName, string outputFileName)
        {
            // Read audio files
            var (audio, fs1) = ReadWaveFile(audioFileName);
            var lpbFileName = audioFileName.Replace("mic.wav", "lpb.wav");
            var (lpb, fs2) = ReadWaveFile(lpbFileName);

            // Check sampling rates
            if (fs1 != SampleRate || fs2 != SampleRate)
                throw new ArgumentException("Sampling rate must be 16kHz.");

            // Check for unequal length and adjust
            if (lpb.shape[0] > audio.shape[0])
                lpb = lpb[":audio.shape[0]"];
            if (lpb.shape[0] < audio.shape[0])
                audio = audio[":lpb.shape[0]"];

            // Save the length of the audio for later
            var lenAudio = audio.shape[0];

            // Pad the audio file
            var padding = np.zeros(BlockLength - BlockShift);
            audio = np.concatenate(new NDarray[] { padding, audio, padding });
            lpb = np.concatenate(new NDarray[] { padding, lpb, padding });

            // Get input/output metadata from sessions
            var inputMetadata1 = _session1.InputMetadata;
            var outputMetadata1 = _session1.OutputMetadata;
            var inputMetadata2 = _session2.InputMetadata;
            var outputMetadata2 = _session2.OutputMetadata;

            // Preallocate states for LSTMs
            var states1Shape = GetTensorShape(inputMetadata1.Values.ElementAt(1));
            var states2Shape = GetTensorShape(inputMetadata2.Values.ElementAt(1));
            var states1 = np.zeros(states1Shape).astype(np.float32);
            var states2 = np.zeros(states2Shape).astype(np.float32);

            // Preallocate output file
            var outFile = np.zeros(audio.shape[0]);

            // Create buffers
            var inBuffer = np.zeros(BlockLength).astype(np.float32);
            var inBufferLpb = np.zeros(BlockLength).astype(np.float32);
            var outBuffer = np.zeros(BlockLength).astype(np.float32);

            // Calculate number of frames
            var numBlocks = (audio.shape[0] - (BlockLength - BlockShift)) / BlockShift;

            // Iterate over the number of frames
            for (int idx = 0; idx < numBlocks; idx++)
            {
                // Shift values and write to buffer of the input audio
                inBuffer[$":-{BlockShift}"] = inBuffer[$"{BlockShift}:"];
                inBuffer[$"-{BlockShift}:"] = audio[$"{idx * BlockShift}:{idx * BlockShift + BlockShift}"];

                // Shift values and write to buffer of the loopback audio
                inBufferLpb[$":-{BlockShift}"] = inBufferLpb[$"{BlockShift}:"];
                inBufferLpb[$"-{BlockShift}:"] = lpb[$"{idx * BlockShift}:{idx * BlockShift + BlockShift}"];

                // Calculate FFT of input block
                var inBlockFft = np.fft.rfft(inBuffer.squeeze()).astype(np.complex64);

                // Create magnitude
                var inMag = np.abs(inBlockFft);
                inMag = np.reshape(inMag, new int[] { 1, 1, -1 }).astype(np.float32);

                // Calculate log pow of lpb
                var lpbBlockFft = np.fft.rfft(inBufferLpb.squeeze()).astype(np.complex64);
                var lpbMag = np.abs(lpbBlockFft);
                lpbMag = np.reshape(lpbMag, new int[] { 1, 1, -1 }).astype(np.float32);

                // Create input tensors for first model
                var inputTensors1 = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(inputMetadata1.Keys.ElementAt(0),
                        CreateTensorFromNDarray(inMag)),
                    NamedOnnxValue.CreateFromTensor(inputMetadata1.Keys.ElementAt(1),
                        CreateTensorFromNDarray(states1)),
                    NamedOnnxValue.CreateFromTensor(inputMetadata1.Keys.ElementAt(2),
                        CreateTensorFromNDarray(lpbMag))
                };

                // Run first model
                using var results1 = _session1.Run(inputTensors1);

                // Get the output of the first block
                var outMaskTensor = results1.ElementAt(0).AsTensor<float>();
                var states1Tensor = results1.ElementAt(1).AsTensor<float>();

                var outMask = TensorToNDarray(outMaskTensor);
                states1 = TensorToNDarray(states1Tensor);

                // Apply mask and calculate the IFFT
                var estimatedBlock = np.fft.irfft(inBlockFft * outMask);

                // Reshape the time domain frames
                estimatedBlock = np.reshape(estimatedBlock, new int[] { 1, 1, -1 }).astype(np.float32);
                var inLpb = np.reshape(inBufferLpb, new int[] { 1, 1, -1 }).astype(np.float32);

                // Create input tensors for second model
                var inputTensors2 = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(inputMetadata2.Keys.ElementAt(0),
                        CreateTensorFromNDarray(estimatedBlock)),
                    NamedOnnxValue.CreateFromTensor(inputMetadata2.Keys.ElementAt(1),
                        CreateTensorFromNDarray(states2)),
                    NamedOnnxValue.CreateFromTensor(inputMetadata2.Keys.ElementAt(2),
                        CreateTensorFromNDarray(inLpb))
                };

                // Run second model
                using var results2 = _session2.Run(inputTensors2);

                // Get output tensors
                var outBlockTensor = results2.ElementAt(0).AsTensor<float>();
                var states2Tensor = results2.ElementAt(1).AsTensor<float>();

                var outBlock = TensorToNDarray(outBlockTensor);
                states2 = TensorToNDarray(states2Tensor);

                // Shift values and write to buffer
                outBuffer[$":-{BlockShift}"] = outBuffer[$"{BlockShift}:"];
                outBuffer[$"-{BlockShift}:"] = np.zeros(BlockShift);
                outBuffer += outBlock.squeeze();

                // Write block to output file
                outFile[$"{idx * BlockShift}:{idx * BlockShift + BlockShift}"] = outBuffer[$":{BlockShift}"];
            }

            // Cut audio to original length
            var predictedSpeech = outFile[$"{BlockLength - BlockShift}:{BlockLength - BlockShift + lenAudio}"];

            // Check for clipping
            float? maxValue = predictedSpeech.GetData<float>().Max();
            if (maxValue > 1.0)
            {
                predictedSpeech = predictedSpeech / maxValue * 0.99;
            }

            // Write output file
            WriteWaveFile(outputFileName, predictedSpeech.GetData<float>(), SampleRate);
        }

        public void ProcessFolder(string inputFolder, string outputFolder)
        {
            var fileNames = new List<string>();
            var directories = new List<string>();
            var newDirectories = new List<string>();

            // Walk through the directory
            foreach (var file in Directory.GetFiles(inputFolder, "*mic.wav", SearchOption.AllDirectories))
            {
                var directory = Path.GetDirectoryName(file);
                var fileName = Path.GetFileName(file);

                fileNames.Add(fileName);
                directories.Add(directory);

                // Create new directory names
                var newDirectory = directory.Replace(inputFolder, outputFolder);
                newDirectories.Add(newDirectory);

                // Check if the new directory already exists, if not create it
                if (!Directory.Exists(newDirectory))
                    Directory.CreateDirectory(newDirectory);
            }

            // Iterate over all .wav files
            for (int idx = 0; idx < fileNames.Count; idx++)
            {
                // Process each file with the model
                ProcessFile(
                    Path.Combine(directories[idx], fileNames[idx]),
                    Path.Combine(newDirectories[idx], fileNames[idx])
                );
                Console.WriteLine($"{fileNames[idx]} processed successfully!");
            }
        }

        private (NDarray audio, int sampleRate) ReadWaveFile(string fileName)
        {
            using var reader = new WaveFileReader(fileName);

            if (reader.WaveFormat.SampleRate != SampleRate)
                throw new ArgumentException($"Expected sample rate {SampleRate}, got {reader.WaveFormat.SampleRate}");

            if (reader.WaveFormat.Channels != 1)
                throw new ArgumentException("Only mono files are supported");

            var samples = new List<float>();
            var buffer = new float[reader.WaveFormat.SampleRate];
            int samplesRead;

            while ((samplesRead = reader.Read([.. buffer.Select(Convert.ToByte)], 0, buffer.Length)) > 0)
            {
                for (int i = 0; i < samplesRead; i++)
                    samples.Add(buffer[i]);
            }

            return (np.array(samples.ToArray()), reader.WaveFormat.SampleRate);
        }

        private void WriteWaveFile(string fileName, float[] audio, int sampleRate)
        {
            var waveFormat = new WaveFormat(sampleRate, 1);
            using var writer = new WaveFileWriter(fileName, waveFormat);
            writer.WriteSamples(audio, 0, audio.Length);
        }

        private int[] GetTensorShape(NodeMetadata nodeMetadata)
        {
            return nodeMetadata.Dimensions.ToArray();
        }

        private Tensor<float> CreateTensorFromNDarray(NDarray ndarray)
        {
            var data = ndarray.GetData<float>();
            var shape = ndarray.shape;
            return new DenseTensor<float>(memory: data, dimensions: shape.Dimensions);
        }

        private NDarray TensorToNDarray(Tensor<float> tensor)
        {
            var data = tensor.ToArray();
            var shape = tensor.Dimensions.ToArray();
            return np.array(data).reshape(shape);
        }

        public void Dispose()
        {
            _session1?.Dispose();
            _session2?.Dispose();
        }
    }
}