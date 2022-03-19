from moviepy.editor import VideoFileClip
from useful_functions import frame_class

## test output video
frm = frame_class()
output_video = 'output_images/test1.mp4'
clip = VideoFileClip('project_video.mp4')
output_clip = clip.fl_image(frm.pipe_line)
output_clip.write_videofile(output_video, audio=False)
