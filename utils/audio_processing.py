import moviepy.editor as mp_editor

def extract_audio(input_video_path, duration, output_audio_path):
    audio = mp_editor.AudioFileClip(input_video_path).subclip(0, duration)
    audio.write_audiofile(output_audio_path)
    return output_audio_path

def merge_audio_with_video(video_path, audio_path, output_video_path):
    video = mp_editor.VideoFileClip(video_path)
    video = video.set_audio(mp_editor.AudioFileClip(audio_path))
    video.write_videofile(output_video_path, codec='libx264')
