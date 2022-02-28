from moviepy.editor import VideoFileClip
from frame_class_file_v2 import frame_class

## test output video
frm = frame_class()
output_video = 'output_images/test1.mp4'
clip = VideoFileClip('project_video.mp4')
output_clip = clip.fl_image(frm.pipe_line)
output_clip.write_videofile(output_video, audio=False)


# self.grayshow(self.warped_compute)
# plt.plot(leftx_in_area_mean, lefty_in_area_mean, 'o')
# plt.plot(rightx_in_area_mean, righty_in_area_mean, 'o')
# lane_width_in_area