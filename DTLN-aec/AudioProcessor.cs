using Emgu.TF.Lite;
using NAudio.Wave;
using Numpy;
using System.Runtime.InteropServices;

namespace DTLN_aec
{
    public class AudioProcessor : IDisposable
    {
        private Interpreter _interpreter1;
        private Interpreter _interpreter2;
        private const int BlockLength = 512;
        private const int BlockShift = 128;
        private const int SampleRate = 16000;

        public AudioProcessor(string modelPath)
        {
            // Load the two model parts
            //var model1Bytes = File.ReadAllBytesAsync($"{modelPath}_1.tflite");
            //var model2Bytes = File.ReadAllBytesAsync($"{modelPath}_2.tflite");
            //Task.WaitAll(model1Bytes, model2Bytes);

            FlatBufferModel model1 = new FlatBufferModel($"{modelPath}_1.tflite");
            FlatBufferModel model2 = new FlatBufferModel($"{modelPath}_2.tflite");

            _interpreter1 = new Interpreter(model1);
            _interpreter1.AllocateTensors();
            _interpreter2 = new Interpreter(model2);
            _interpreter2.AllocateTensors();
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

            // Check for single channel (already handled in ReadWaveFile)

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

            // Get details from interpreters
            var inputDetails1 = _interpreter1.Inputs;
            var outputDetails1 = _interpreter1.Outputs;
            var inputDetails2 = _interpreter2.Inputs;
            var outputDetails2 = _interpreter2.Inputs;

            // Preallocate states for LSTMs
            var states1 = np.zeros(GetTensorShape(inputDetails1[1])).astype(np.float32);
            var states2 = np.zeros(GetTensorShape(inputDetails2[1])).astype(np.float32);

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

                var inMagBytes = inMag.tobytes();
                var state1Bytes = states1.tobytes();
                var lpbMagytes = lpbMag.tobytes();

                // Set tensors to the first model
                Marshal.Copy(inMagBytes, 0, _interpreter2.Inputs[0].DataPointer, inMagBytes.Length);
                Marshal.Copy(state1Bytes, 0, _interpreter2.Inputs[1].DataPointer, state1Bytes.Length);
                Marshal.Copy(lpbMagytes, 0, _interpreter2.Inputs[2].DataPointer, lpbMagytes.Length);

                // Run calculation
                _interpreter1.Invoke();

                // Get the output of the first block
                var outMask = np.array<float>(_interpreter1.Outputs[0]);
                states1 = np.array<float>(_interpreter1.Outputs[1]);

                // Apply mask and calculate the IFFT
                var estimatedBlock = np.fft.irfft(inBlockFft * outMask);

                // Reshape the time domain frames
                estimatedBlock = np.reshape(estimatedBlock, new int[] { 1, 1, -1 }).astype(np.float32);
                var inLpb = np.reshape(inBufferLpb, new int[] { 1, 1, -1 }).astype(np.float32);

                var estimatedBlockBytes = estimatedBlock.tobytes();
                var states2Bytes = states2.tobytes();
                var inLpbBytes = inLpb.tobytes();

                Marshal.Copy(estimatedBlockBytes, 0, _interpreter2.Inputs[0].DataPointer, estimatedBlockBytes.Length);
                Marshal.Copy(states2Bytes, 0, _interpreter2.Inputs[1].DataPointer, states2Bytes.Length);
                Marshal.Copy(inLpbBytes, 0, _interpreter2.Inputs[2].DataPointer, inLpbBytes.Length);

                // Run calculation
                _interpreter2.Invoke();

                // Get output tensors
                var outBlock = np.array<float>(_interpreter2.Outputs[0]);
                states2 = np.array<float>(_interpreter2.Outputs[1]);

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

        private int[] GetTensorShape(Tensor tensor)
        {
            // Return the shape array from tensor info
            return tensor.Dims;
        }

        public void Dispose()
        {
            _interpreter1?.Dispose();
            _interpreter2?.Dispose();
        }
    }
}