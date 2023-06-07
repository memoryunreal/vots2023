# import for debugging
import os
import glob
import numpy as np
from PIL import Image
# import for base_tracker
import torch
import yaml
import torch.nn.functional as F
from model.network import XMem
from inference.inference_core import InferenceCore
from util.mask_mapper import MaskMapper
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from util.range_transform import im_normalization

from tools.painter import mask_painter
import progressbar

import hickle as hkl
import os


class BaseTracker:
	def __init__(self, xmem_checkpoint, device, size=-1, mem_every=5, flip=False) -> None:
		"""
		device: model device
		xmem_checkpoint: checkpoint of XMem model
		"""
		# load configurations
		with open("/home/gaomingqi/methods/vots2023/experiment/VideoX/tracker-ms/config/config.yaml", 'r') as stream: 
			config = yaml.safe_load(stream)
		
		config['mem_every'] = mem_every
		config['size'] = size
		config['flip'] = flip

		# initialise XMem
		network = XMem(config, xmem_checkpoint, map_location='cpu').to(device).eval()
		# initialise IncerenceCore
		self.tracker = InferenceCore(network, config)
		# data transformation
		if size < 0:
			self.im_transform = transforms.Compose([
				transforms.ToTensor(),
				im_normalization,
			])
			self.need_resize = False
		else:
			self.im_transform = transforms.Compose([
				transforms.ToTensor(),
				im_normalization,
				transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
			])
			self.need_resize = True
		self.device = device
		self.size = size
		self.flip = flip
		self.mem_every = mem_every

		# changable properties
		self.mapper = MaskMapper()
		self.initialised = False


	@torch.no_grad()
	def resize_mask(self, mask):
		# mask transform is applied AFTER mapper, so we need to post-process it in eval.py
		h, w = mask.shape[-2:]
		min_hw = min(h, w)
		return F.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)), 
					mode='nearest')

	@torch.no_grad()
	def track(self, frame, first_frame_annotation=None):
		"""
		Input: 
		frames: numpy arrays (H, W, 3)
		logit: numpy array (H, W), logit

		Output:
		mask: numpy arrays (H, W)
		logit: numpy arrays, probability map (H, W)
		painted_image: numpy array (H, W, 3)
		"""
		shape = frame.shape[:2]

		if first_frame_annotation is not None:   # first frame mask
			# initialisation
			mask, labels = self.mapper.convert_mask(first_frame_annotation)
			mask = torch.Tensor(mask).to(self.device)
			# flip and resize the first frame annotation
			if self.flip:
				mask = torch.flip(mask, dims=[-1])
			if self.need_resize:
				mask = self.resize_mask(mask.unsqueeze(0))[0]
			self.tracker.set_all_labels(list(self.mapper.remappings.values()))
		else:
			mask = None
			labels = None
		
		# prepare inputs, resize and flip
		frame_tensor = self.im_transform(frame).to(self.device)
		if self.flip:
			frame_tensor = torch.flip(frame_tensor, dims=[-1])

		# track one frame
		probs, _ = self.tracker.step(frame_tensor, mask, labels)   # logits 2 (bg fg) H W

		# resize and flip back to original
		if self.need_resize:
			probs = F.interpolate(probs.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]
		if self.flip:
			probs = torch.flip(probs, dims=[-1])

		# # convert to mask
		# out_mask = torch.argmax(probs, dim=0)
		# out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
		# final_mask = np.zeros_like(out_mask)
		
		# for ms inference
		probs = (probs.detach().cpu().numpy()*255).astype(np.uint8)
		return probs

		# # map back
		# for k, v in self.mapper.remappings.items():
		# 	final_mask[out_mask == v] = k

		# num_objs = final_mask.max()
		# painted_image = frame
		# for obj in range(1, num_objs+1):
		# 	if np.max(final_mask==obj) == 0:
		# 		continue
		# 	painted_image = mask_painter(painted_image, (final_mask==obj).astype('uint8'), mask_color=obj+1)

		# # print(f'max memory allocated: {torch.cuda.max_memory_allocated()/(2**20)} MB')

		# return final_mask, final_mask, painted_image

	@torch.no_grad()
	def clear_memory(self):
		self.tracker.clear_memory()
		self.mapper.clear_labels()
		torch.cuda.empty_cache()


class MSTracker:
	def __init__(self, xmem_checkpoint, device_list, scale_list):
		"""
		  device_list: [1, 2, 3, ...]
		  scale_list:[
		  {'size': s1, 'mem_every': m1, 'flip': f1}, 
		  {'size': s2, 'mem_every': m2, 'flip': f2}, 
		  {'size': s3, 'mem_every': m3, 'flip': f3}, 
		  ...
		  ]
		"""
		self.tracker_list = []
		self.num_trackers = 0
		for device, scale in zip(device_list, scale_list):
			self.tracker_list.append(BaseTracker(xmem_checkpoint, device, scale['size'], scale['mem_every'], scale['flip']))
			self.num_trackers += 1
	
	@torch.no_grad()
	def track(self, frame, first_frame_annotation=None):
		result_sum = None
		for tracker in self.tracker_list:
			# tracker.track returns: probs
			result = tracker.track(frame, first_frame_annotation)
			if result_sum is None:
				result_sum = result.astype(np.float32)
			else:
				result_sum += result

		# argmax and to idx
		result_sum = np.argmax(result_sum, axis=0)
		
		# convert to mask
		final_mask = np.zeros_like(result_sum, dtype=np.uint8)
		for k, v in self.tracker_list[0].mapper.remappings.items():
			final_mask[result_sum == v] = k
		
		num_objs = final_mask.max()
		painted_frame = frame
		for obj in range(1, num_objs+1):
			if np.max(final_mask==obj) == 0:
				continue
			painted_frame = mask_painter(painted_frame, (final_mask==obj).astype('uint8'), mask_color=obj+1)

		return final_mask, painted_frame

	@torch.no_grad()
	def clear_memory(self):
		for tracker in self.tracker_list:
			tracker.clear_memory()


if __name__ == '__main__':
	# load videos from davis-2017-val
	f = open('/ssd1/gaomingqi/datasets/davis/ImageSets/2017/val.txt')
	vid_list = f.read().splitlines()
	# for each video
	for vid in progressbar.progressbar(vid_list):
		frame_list = glob.glob(os.path.join('/ssd1/gaomingqi/datasets/davis/JPEGImages/480p/', vid, '*.jpg'))
		frame_list.sort()
	
		# load frames
		frames = []
		for frame_path in frame_list:
			frames.append(np.array(Image.open(frame_path).convert('RGB')))
		frames = np.stack(frames, 0)    # T, H, W, C
		
		# load first frame annotation
		first_frame_path = os.path.join('/ssd1/gaomingqi/datasets/davis/Annotations/480p/', vid, '00000.png')
		first_frame_annotation = np.array(Image.open(first_frame_path).convert('P'))    # H, W, C

		# ************************************************************************************
		# ------------------------------------------------------------------------------------
		# how to use
		# ------------------------------------------------------------------------------------
		# 1/5: set checkpoint
		XMEM_checkpoint = '/ssd1/gaomingqi/checkpoints/XMem-s012.pth'
		# 2/5: initialise devices and scale data
		device_list = [3, 4]
		scale_list = [
			{'size': 720, 'mem_every': 3, 'flip': False},
			{'size': 720, 'mem_every': 3, 'flip': True},
			]
		# ------------------------------------------------------------------------------------
		# 3/5: initialise tracker
		mstracker = MSTracker(XMEM_checkpoint, device_list, scale_list)
		# ------------------------------------------------------------------------------------
		# 4/5: for each frame, get tracking results by tracker.track(frame, first_frame_annotation)
		# frame: numpy array (H, W, C), first_frame_annotation: numpy array (H, W), leave it blank when tracking begins
		painted_frames = []
		masks = []
		for ti, frame in enumerate(frames):
			if ti == 0:
				final_mask, painted_frame = mstracker.track(frame, first_frame_annotation)
			else:
				final_mask, painted_frame = mstracker.track(frame)
			painted_frames.append(painted_frame)
			masks.append(final_mask)
		# ------------------------------------------------------------------------------------
		# 5/5: clear memory in XMEM for the next video
		mstracker.clear_memory()
		# ------------------------------------------------------------------------------------
		# end
		# ------------------------------------------------------------------------------------
		# ************************************************************************************

		# set saving path
		save_path = os.path.join('/ssd1/gaomingqi/results/TAM/ms-davis-17-val', vid)
		if not os.path.exists(save_path):
			os.mkdir(save_path)
		# save
		for ti, mask in enumerate(masks):
			mask = Image.fromarray(mask)
			mask.save(f'{save_path}/{ti:05d}.png')
