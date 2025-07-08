import librosa
import noisereduce as nr
import soundfile as sf

# 1. 오디오 불러오기
y, sr = librosa.load("your_audio.wav", sr=22050)

# 2. 앞부분 0.5초를 '노이즈 프로파일'로 설정 (조용한 구간이면 좋음)
noise_clip = y[:int(0.5 * sr)]

# 3. 노이즈 제거 (prop_decrease로 제거 정도 조절)
y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip, prop_decrease=0.8)

# 4. 비교용으로 저장
sf.write("denoised_08.wav", y_denoised, sr)
