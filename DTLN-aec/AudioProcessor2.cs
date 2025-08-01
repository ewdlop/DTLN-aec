using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using NAudio.Wave;
using System.Numerics;

namespace DTLN_aec
{
    public class AudioProcessor2 : IDisposable
    {
        private readonly InferenceSession _session1;
        private readonly InferenceSession _session2;
        private const int BlockLength = 512;
        private const int BlockShift = 128;
        private const int SampleRate = 16000;

        public AudioProcessor(string modelPath)
        {
            // 載入已轉成 ONNX 的兩個子模型
            _session1 = new InferenceSession($"{modelPath}_1.onnx");
            _session2 = new InferenceSession($"{modelPath}_2.onnx");
        }

        public void ProcessFile(string audioFileName, string outputFileName)
        {
            // 讀檔
            var (audio, fs1) = ReadWaveFile(audioFileName);
            var (lpb, fs2) = ReadWaveFile(audioFileName.Replace("mic.wav", "lpb.wav"));

            if (fs1 != SampleRate || fs2 != SampleRate)
                throw new ArgumentException("Sampling rate must be 16kHz.");

            // 對齊長度
            int len = Math.Min(audio.Length, lpb.Length);
            Array.Resize(ref audio, len);
            Array.Resize(ref lpb, len);

            // 前後各 pad
            var pad = new float[BlockLength - BlockShift];
            audio = pad.Concat(audio).Concat(pad).ToArray();
            lpb = pad.Concat(lpb).Concat(pad).ToArray();

            int nBins = BlockLength / 2 + 1;
            int numBlocks = (audio.Length - (BlockLength - BlockShift)) / BlockShift;

            // 狀態張量（以 1×1×… 形式輸入）
            var state1 = new float[_session1.InputMetadata.ElementAt(1).Value.Dimensions.Aggregate(1, (a, b) => a * b)];
            var state2 = new float[_session2.InputMetadata.ElementAt(1).Value.Dimensions.Aggregate(1, (a, b) => a * b)];

            var outFile = new float[audio.Length];
            var inBuf = new float[BlockLength];
            var inLpb = new float[BlockLength];
            var outBuf = new float[BlockLength];

            for (int i = 0; i < numBlocks; i++)
            {
                // 滑動窗口
                Array.Copy(inBuf, BlockShift, inBuf, 0, BlockLength - BlockShift);
                Array.Copy(inLpb, BlockShift, inLpb, 0, BlockLength - BlockShift);
                Array.Copy(audio, i * BlockShift, inBuf, BlockLength - BlockShift, BlockShift);
                Array.Copy(lpb, i * BlockShift, inLpb, BlockLength - BlockShift, BlockShift);

                // FFT
                var bufferC = inBuf.Select(v => (Complex)v).ToArray();
                Fourier.Forward(bufferC, FourierOptions.Matlab);
                var mag1 = bufferC.Take(nBins).Select(c => (float)c.Magnitude).ToArray();

                var bufferL = inLpb.Select(v => (Complex)v).ToArray();
                Fourier.Forward(bufferL, FourierOptions.Matlab);
                var mag2 = bufferL.Take(nBins).Select(c => (float)c.Magnitude).ToArray();

                // 建立輸入張量
                var tensor1 = new DenseTensor<float>(mag1, new[] { 1, 1, nBins });
                var tensorState1 = new DenseTensor<float>(state1, _session1.InputMetadata.ElementAt(1).Value.Dimensions);
                var tensor2InLpb = new DenseTensor<float>(mag2, new[] { 1, 1, nBins });

                // 呼叫第一階段模型
                using var inputs1 = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(_session1.InputMetadata.ElementAt(0).Key, tensor1),
                    NamedOnnxValue.CreateFromTensor(_session1.InputMetadata.ElementAt(1).Key, tensorState1),
                    NamedOnnxValue.CreateFromTensor(_session1.InputMetadata.ElementAt(2).Key, tensor2InLpb),
                };
                using var results1 = _session1.Run(inputs1);
                var mask = results1.First(r => r.Name == _session1.OutputMetadata.ElementAt(0).Key).AsTensor<float>().ToArray();
                state1 = results1.First(r => r.Name == _session1.OutputMetadata.ElementAt(1).Key).AsTensor<float>().ToArray();

                // 反 FFT 還原語音
                var estimated = new Complex[BlockLength];
                var blockC = bufferC; // 原始 FFT 結果
                for (int k = 0; k < nBins; k++)
                    blockC[k] *= mask[k];
                // Hermitian symmetry
                for (int k = nBins; k < BlockLength; k++)
                    blockC[k] = Complex.Conjugate(blockC[BlockLength - k]);
                Fourier.Inverse(blockC, FourierOptions.Matlab);
                for (int j = 0; j < BlockLength; j++)
                    estimated[j] = blockC[j];

                // 第二階段模型：把前面處理結果當作特徵，再跑一次
                var estMag = estimated.Select(c => (float)c.Real).ToArray();
                var tensorEst = new DenseTensor<float>(estMag, new[] { 1, 1, BlockLength });
                var tensorState2 = new DenseTensor<float>(state2, _session2.InputMetadata.ElementAt(1).Value.Dimensions);

                using var inputs2 = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(_session2.InputMetadata.ElementAt(0).Key, tensorEst),
                    NamedOnnxValue.CreateFromTensor(_session2.InputMetadata.ElementAt(1).Key, tensorState2),
                    // 如果第二階段需要額外輸入，可同理加入
                };
                using var results2 = _session2.Run(inputs2);
                var outBlock = results2.First(r => r.Name == _session2.OutputMetadata.ElementAt(0).Key).AsTensor<float>().ToArray();
                state2 = results2.First(r => r.Name == _session2.OutputMetadata.ElementAt(1).Key).AsTensor<float>().ToArray();

                // Overlap–add
                Array.Copy(outBuf, BlockShift, outBuf, 0, BlockLength - BlockShift);
                for (int j = 0; j < BlockShift; j++) outBuf[BlockLength - BlockShift + j] = 0;
                for (int j = 0; j < BlockLength; j++) outBuf[j] += outBlock[j];

                // 寫回最終輸出
                Array.Copy(outBuf, 0, outFile, i * BlockShift, BlockShift);
            }

            // 裁切到原長度並寫檔
            var final = new float[audio.Length - 2 * (BlockLength - BlockShift)];
            Array.Copy(outFile, BlockLength - BlockShift, final, 0, final.Length);
            WriteWaveFile(outputFileName, final, SampleRate);
        }

        private (float[] data, int sr) ReadWaveFile(string path)
        {
            using var rdr = new WaveFileReader(path);
            if (rdr.WaveFormat.SampleRate != SampleRate || rdr.WaveFormat.Channels != 1)
                throw new ArgumentException("只支援 16kHz 單聲道");
            var samples = new List<float>();
            var buf = new float[1024];
            int read;
            while ((read = rdr.Read(buf, 0, buf.Length)) > 0)
                samples.AddRange(buf.Take(read));
            return (samples.ToArray(), rdr.WaveFormat.SampleRate);
        }

        private void WriteWaveFile(string path, float[] data, int sr)
        {
            using var wr = new WaveFileWriter(path, new WaveFormat(sr, 1));
            wr.WriteSamples(data, 0, data.Length);
        }

        public void Dispose()
        {
            _session1.Dispose();
            _session2.Dispose();
        }
    }
}
