#!/usr/bin/env python
"""
Deep neural network that detects and gives the coordinates and sizes of all the occurrences of a desired object in a picture.
Use a deep convolutional network pre-trained on ImageNet for the feature extraction.
Train the classification and regression using the dataset COCO.

Download the datasets:

%%bash -s "$dataset_dir"

tmp_path=$1
ds_path=$1

if [ ! -d $ds_path ]; then
	for ds in train2017 val2017 test2017 annotations_trainval2017 ; do
		wget http://images.cocodataset.org/zips/$ds.zip -P $tmp_path
		unzip -qd $ds_path/ $tmp_path/$ds.zip
		rm $tmp_path/$ds.zip
	done
fi

Author: Arthur Bouton [arthur.bouton@gadz.org]

"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
import pandas as pd
import random
import sys
import os
#from looptools import *
#import signal
#signal.signal( signal.SIGINT, signal.SIG_DFL )


target_label = 'bird'

dataset_dir = './datasets/coco'
session_dir = './training_data/cnn_multi_targets/v1'

os.makedirs( session_dir + '/checkpoints', exist_ok=True )



#########
# Model #
#########

input_w = 224
input_h = 224


from tensorflow.keras.applications.xception import Xception, preprocess_input

xception_model = Xception( weights='imagenet', include_top=False, input_shape=( input_h, input_w, 3 ) )

grid_w = xception_model.output.shape[1]
grid_h = xception_model.output.shape[2]

if len( sys.argv ) > 1 and sys.argv[1] == 'pretrain' :
	for layer in xception_model.layers:
		layer.trainable = False


from tensorflow.keras import layers, activations

reg = tf.keras.regularizers.l2( 0.0005 )


x = xception_model.output

for _ in range( 3 ) :
	x = layers.Conv2D( kernel_size=3, filters=1024, strides=1, padding='same', kernel_regularizer=reg )( x )
	x = layers.BatchNormalization()( x )
	x = layers.LeakyReLU( 0.1 )( x )

x = layers.Conv2D( kernel_size=1, filters=4, strides=1, padding='same' )( x )

model = tf.keras.Model( xception_model.input, x )

#model.load_weights( session_dir + '/checkpoints/ckpt' ).expect_partial()


#print( 'Number of parameters in the model: %i' % model.count_params() )
#tf.keras.utils.plot_model( model, show_shapes=True )
#exit( 0 )



##################
# Custom metrics #
##################

def detection_loss( y_true, y_pred ) :
	partitions = tf.cast( y_true[:,:,:,0] == 1, tf.int32 )
	y_true_absent, y_true_present = tf.dynamic_partition( y_true, partitions, 2 )
	y_pred_absent, y_pred_present = tf.dynamic_partition( y_pred, partitions, 2 )

	coefs = tf.constant( [ 1, 1, 1, 1 ], dtype=tf.float32 )
	present_loss = tf.reduce_mean( tf.reduce_sum( tf.square( y_true_present - y_pred_present )*coefs, 1 ) )
	#present_loss = tf.reduce_mean( tf.reduce_sum( -tf.math.log( 1 - tf.abs( y_true_present - y_pred_present ) )*coefs, 1 ) )

	absent_loss = tf.reduce_mean( tf.square( y_pred_absent[:,0] ) )
	#absent_loss = tf.reduce_mean( -tf.math.log( 1 - y_pred_absent[:,0] ) )

	return present_loss + absent_loss

def accuracy( y_true, y_pred ) :
	threshold = 0.5
	d_pred = y_pred[:,:,:,0]
	d_true = y_true[:,:,:,0]
	true_positives = tf.math.count_nonzero( tf.logical_and( d_pred > threshold, d_true > threshold ), dtype=tf.float32 )
	true_negatives = tf.math.count_nonzero( tf.logical_and( d_pred < threshold, d_true < threshold ), dtype=tf.float32 )
	return ( true_positives + true_negatives )/tf.cast( len( d_pred )*d_pred.shape[1]*d_pred.shape[2], tf.float32 )

def precision( y_true, y_pred ) :
	threshold = 0.5
	d_pred = y_pred[:,:,:,0]
	d_true = y_true[:,:,:,0]
	true_positives = tf.math.count_nonzero( tf.logical_and( d_pred > threshold, d_true > threshold ), dtype=tf.float32 )
	if true_positives == 0 :
		return 0.
	false_positives = tf.math.count_nonzero( tf.logical_and( d_pred > threshold, d_true < threshold ), dtype=tf.float32 )
	return true_positives/( true_positives + false_positives )

def recall( y_true, y_pred ) :
	threshold = 0.5
	d_pred = y_pred[:,:,:,0]
	d_true = y_true[:,:,:,0]
	true_positives = tf.math.count_nonzero( tf.logical_and( d_pred > threshold, d_true > threshold ), dtype=tf.float32 )
	if true_positives == 0 :
		return 0.
	false_negatives = tf.math.count_nonzero( tf.logical_and( d_pred < threshold, d_true > threshold ), dtype=tf.float32 )
	return true_positives/( true_positives + false_negatives )

def Fscore( y_true, y_pred ) :
	threshold = 0.5
	d_pred = y_pred[:,:,:,0]
	d_true = y_true[:,:,:,0]
	true_positives = tf.math.count_nonzero( tf.logical_and( d_pred > threshold, d_true > threshold ), dtype=tf.float32 )
	if true_positives == 0 :
		return 0.
	false_positives = tf.math.count_nonzero( tf.logical_and( d_pred > threshold, d_true < threshold ), dtype=tf.float32 )
	false_negatives = tf.math.count_nonzero( tf.logical_and( d_pred < threshold, d_true > threshold ), dtype=tf.float32 )
	precision = true_positives/( true_positives + false_positives )
	recall = true_positives/( true_positives + false_negatives )
	return 2*precision*recall/( precision + recall )

def aiming( y_true, y_pred ) :
	partitions = tf.cast( y_true[:,:,:,0] == 1, tf.int32 )
	_, y_true_present = tf.dynamic_partition( y_true, partitions, 2 )
	_, y_pred_present = tf.dynamic_partition( y_pred, partitions, 2 )

	ratio = tf.constant( [ input_w/grid_w, input_h/grid_h ], dtype=tf.float32 )
	distances = tf.sqrt( tf.reduce_sum( tf.square( ( y_true_present[:,1:3] - y_pred_present[:,1:3] )*ratio ), 1 ) )
	return tf.reduce_mean( distances )



########
# Test #
########

if len( sys.argv ) > 1 and sys.argv[1] == 'test' :

	if len( sys.argv ) > 2 :
		threshold = float( sys.argv[2] )
	else :
		threshold = 0.9

	#model = tf.keras.models.load_model( session_dir + '/model', custom_objects={ 'detection_loss': detection_loss, 'accuracy': accuracy, 'aiming': aiming } )
	model.load_weights( session_dir + '/checkpoints/ckpt' ).expect_partial()
	
	import glob

	try :
		for file_path in glob.glob( 'datasets/test/*.jpg' ) :
		#for file_path in glob.glob( dataset_dir + '/test2017/*.jpg' ) :

			# Read the image from its file:
			image = tf.keras.preprocessing.image.load_img( file_path )
			image = tf.keras.preprocessing.image.img_to_array( image )
			image = tf.image.resize_with_pad( image, input_h, input_w )

			# Display the image:
			plt.imshow( image/255 )

			# Feed the image to the model:
			#image = tf.image.per_image_standardization( image )
			#image = image/127.5 - 1
			image = preprocess_input( image )
			output = tf.squeeze( model( tf.expand_dims( image, 0 ) ) )

			# Draw the grid:
			for i in range( 1, grid_w ) :
				plt.axvline( i/grid_w*input_w, c='b' )
			for j in range( 1, grid_h ) :
				plt.axhline( j/grid_h*input_h, c='b' )
			
			# Draw the reticles:
			print( 'File: %s' % file_path )
			for i in range( grid_w ) :
				for j in range( grid_h ) :
					if output[i,j,0] > threshold :
						print( 'Cell %i,%i:' % ( i + 1, j + 1 ), output[i,j,:].numpy() )
						x = ( output[i,j,1] + i )/grid_w*input_w
						y = ( output[i,j,2] + j )/grid_h*input_h
						reticle_color = 'r'
						plt.axvline( x, c=reticle_color )
						plt.axhline( y, c=reticle_color )
						if output[i,j,3] > 0 :
							r = output[i,j,3]*max( input_w, input_h )/2
							plt.gcf().gca().add_artist( plt.Circle( ( x, y ), r, fill=False, color=reticle_color ) )

			plt.show()

	except KeyboardInterrupt :
		pass
	print()
	exit( 0 )



########################
# Prepare the datasets #
########################

def prepare_dataset( instance_file, image_dir, target_label, bb_ratio_min=0.1, bb_ratio_max=1, randomize=True ) :

	print( 'Extracting the annotations from %s' % instance_file )

	with open( instance_file ) as json_file :
		data = json.load( json_file )

	image_df = pd.DataFrame( data['images'] )
	annotation_df = pd.DataFrame( data['annotations'] )


	target_id = [ d for d in data['categories'] if d['name'] == target_label ][0]['id']

	# IDs of all the images containing the desired object:
	present_image_ids = annotation_df[ annotation_df.category_id == target_id ].image_id.unique()
	# IDs of all the images that don't contain the desired object:
	absent_image_ids = image_df[ ~image_df.id.isin( present_image_ids ) ].id.unique()

	# Remove the IDs of the image containing a crowd of the desired object:
	crowd_ids = annotation_df[ ( annotation_df.category_id == target_id ) & ( annotation_df.iscrowd == 1 ) ].image_id.unique()
	present_image_ids = annotation_df[ ( annotation_df.category_id == target_id ) & ( ~annotation_df.image_id.isin( crowd_ids ) ) ].image_id.unique()


	print( 'Number of images containing at least one %s: %6i' % ( target_label, len( present_image_ids ) ) )
	print( 'Number of images containing no %s:           %6i' % ( target_label, len( absent_image_ids ) ) )


	#present_image_ids = present_image_ids[:5]
	#absent_image_ids = absent_image_ids[:5]


	def gen_dataset() :

		while True :

			if randomize :
				np.random.shuffle( present_image_ids )
				np.random.shuffle( absent_image_ids )

			for interleaved_image_ids in zip( present_image_ids, absent_image_ids ) :
				for image_id in interleaved_image_ids :

					image_data = image_df[ image_df.id == image_id ]

					# Read the image from its file:
					image = tf.keras.preprocessing.image.load_img( image_dir + image_data.file_name.iloc[0] )
					image = tf.keras.preprocessing.image.img_to_array( image )

					initial_w = image_data.width.iloc[0]
					initial_h = image_data.height.iloc[0]

					if initial_w < input_w or initial_h < input_h :
						initial_w = max( initial_w, input_w )
						initial_h = max( initial_h, input_h )
						image = tf.image.resize_with_pad( image, initial_h, initial_w ).numpy()
					
					# List the bounding boxes corresponding to the desired category:
					bboxes = annotation_df[ ( annotation_df.image_id == image_id ) & ( annotation_df.category_id == target_id ) ].bbox

					# If there is at least one desired object in the image:
					if bboxes.size > 0 :

						# Compute the center coordinates and size of each object:
						objects = [ ( x + w/2, y + h/2, np.sqrt( w*h ) ) for x, y, w, h in bboxes ]

						# Select one of the object as a reference for cropping:
						if randomize :
							ref_x, ref_y, ref_s = random.choice( objects )
						else :
							ref_x, ref_y, ref_s = objects[0]

						# Choose a reduction ratio for the crop:
						reduction_low_bound = max( 1, ref_s/( input_w*bb_ratio_max ),
													  ref_s/( input_h*bb_ratio_max ) )
						reduction_high_bound = min( ref_s/( input_w*bb_ratio_min ), initial_w/input_w,
													ref_s/( input_h*bb_ratio_min ), initial_h/input_h )
						if not randomize or reduction_high_bound < reduction_low_bound :
							reduction = max( 1, reduction_high_bound )
						else :
							reduction = np.random.uniform( reduction_low_bound, reduction_high_bound )

						# Size of the crop:
						crop_w = int( input_w*reduction )
						crop_h = int( input_h*reduction )

						# Choose the position of the crop:
						if randomize :
							crop_x = int( np.random.uniform( max( 0, ref_x - crop_w ), min( ref_x, initial_w - crop_w ) ) )
							crop_y = int( np.random.uniform( max( 0, ref_y - crop_h ), min( ref_y, initial_h - crop_h ) ) )
						else :
							crop_x = int( ( max( 0, ref_x - crop_w ) + min( ref_x, initial_w - crop_w ) )/2 )
							crop_y = int( ( max( 0, ref_y - crop_h ) + min( ref_y, initial_h - crop_h ) )/2 )

						# Update and normalize the coordinates and sizes of the objects according to the crop:
						objects = [ [ ( o[0] - crop_x )/crop_w, ( o[1] - crop_y )/crop_h, o[2]/max( crop_w, crop_h ) ] for o in objects ]

						# Crop and resize the image:
						image = tf.image.crop_to_bounding_box( image, crop_y, crop_x, crop_h, crop_w )
						image = tf.image.resize_with_pad( image, input_h, input_w )

						# Add random transformations to the image:
						flipped = False
						angle = 0
						if randomize :
							# Randomly flip the image:
							if np.random.randint( 0, 2 ) :
								image = tf.image.flip_left_right( image )
								flipped = True

							# Randomly rotate the image:
							rot_max = 20
							angle = np.random.uniform( -rot_max, rot_max )
							image = tf.keras.preprocessing.image.apply_affine_transform( image.numpy(), angle, fill_mode='reflect' )

							# Randomly modify the brightness of the image:
							delta = np.random.uniform( -0.1, 0.1 )
							image = tf.image.adjust_brightness( image, delta*255 )

							# Randomly modify the saturation of the image:
							factor = np.random.uniform( 0.5, 1.5 )
							image = tf.image.adjust_saturation( image, factor )

						# Normalize the image:
						#image = tf.image.per_image_standardization( image )
						#image = image/127.5 - 1
						image = preprocess_input( image )


						# Compute the targets:
						target = np.zeros( ( grid_w, grid_h, 4 ) )

						for obj in objects :
							# Update the object coordinates according to the image transformations:
							if flipped :
								obj[0] = 1 - obj[0]
							if angle != 0 :
								cosa = np.cos( angle*np.pi/180 )
								sina = np.sin( angle*np.pi/180 )
								new_coords = np.array([ [ cosa, -sina ], [ sina, cosa ] ])@( np.array( obj[:2] ) - 0.5 ) + 0.5
								obj[:2] = new_coords
							# Skip the objects that are outside the crop:
							if not 0 <= obj[0] < 1 or not 0 <= obj[1] < 1 :
								continue
							# Determine the cell corresponding to the object coordinates:
							i = int( grid_w*obj[0] )
							j = int( grid_h*obj[1] )
							# If several objects are in the same cell, keep only the largest one:
							if obj[2] > target[i,j,3] :
								target[i,j,0] = 1
								target[i,j,1] = obj[0]*grid_w - i
								target[i,j,2] = obj[1]*grid_h - j
								target[i,j,3] = obj[2]


						yield image, target

					else :
						if randomize :
							# Randomly rotate the image:
							rot_max = 20
							angle = np.random.uniform( -rot_max, rot_max )
							image = tf.keras.preprocessing.image.apply_affine_transform( image, angle, fill_mode='reflect' )

							reduction = np.random.uniform( 1, min( initial_w/input_w, initial_h/input_h ) )

							crop_w = int( input_w*reduction )
							crop_h = int( input_h*reduction )

							crop_x = int( np.random.uniform( 0, initial_w - crop_w ) )
							crop_y = int( np.random.uniform( 0, initial_h - crop_h ) )

							# Randomly crop the image:
							image = tf.image.crop_to_bounding_box( image, crop_y, crop_x, crop_h, crop_w )

						# Resize the image:
						image = tf.image.resize_with_pad( image, input_h, input_w )

						# Add random transformations to the image:
						if randomize :
							if np.random.randint( 0, 2 ) :
								image = tf.image.flip_left_right( image )

							# Randomly modify the brightness of the image:
							delta = np.random.uniform( -0.1, 0.1 )
							image = tf.image.adjust_brightness( image, delta*255 )

							# Randomly modify the saturation of the image:
							factor = np.random.uniform( 0.5, 1.5 )
							image = tf.image.adjust_saturation( image, factor )

						# Normalize the image:
						#image = tf.image.per_image_standardization( image )
						#image = image/127.5 - 1
						image = preprocess_input( image )


						yield image, tf.zeros( ( grid_w, grid_h, 4 ) )

	#output_signature = ( tf.TensorSpec( shape=( input_h, input_w, 3 ), dtype=tf.float32 ),
                         #tf.TensorSpec( shape=( 7, 7, 3 ), dtype=tf.float32 ) )
	#return tf.data.Dataset.from_generator( gen_dataset, output_signature=output_signature )
	return tf.data.Dataset.from_generator( gen_dataset, output_types=( tf.float32, tf.float32 ),
	                                                    output_shapes=( ( input_h, input_w, 3 ), ( grid_w, grid_h, 4 ) ) )

ds_train = prepare_dataset( dataset_dir + '/annotations/instances_train2017.json',
                            dataset_dir + '/train2017/', target_label, 0.2, 0.5 )
ds_val   = prepare_dataset( dataset_dir + '/annotations/instances_val2017.json',
                            dataset_dir + '/val2017/', target_label, 0.2, 0.5, randomize=False )


if len( sys.argv ) > 1 and sys.argv[1] == 'verify' :

	# Verify the dataset generated:
	try :
		count = 0
		for image, target in ds_train :
		#for image, target in ds_val :

			count += 1
			print( '\rCount:', count, end='', flush=True )

			#if target[0] < 0 :
				#continue

			#print( '\nTarget:', target.numpy() )

			# Display the image:
			image = tf.clip_by_value( ( image + 1 )/2, 0, 1 )
			plt.imshow( image )
			
			# Draw the reticles:
			for i in range( grid_w ) :
				for j in range( grid_h ) :
					if target[i,j,0] > 0.5 :
						x = ( target[i,j,1] + i )/grid_w*input_w
						y = ( target[i,j,2] + j )/grid_h*input_h
						reticle_color = 'r'
						plt.axvline( x, c=reticle_color )
						plt.axhline( y, c=reticle_color )
						if target[i,j,3] > 0 :
							r = target[i,j,3]*max( input_w, input_h )/2
							plt.gcf().gca().add_artist( plt.Circle( ( x, y ), r, fill=False, color=reticle_color ) )

			plt.show()

	except KeyboardInterrupt :
		pass
	print()
	exit( 0 )



##############
# Evaluation #
##############

if len( sys.argv ) > 1 and sys.argv[1] == 'eval' :

	#model = tf.keras.models.load_model( session_dir + '/model', custom_objects={ 'detection_loss': detection_loss, 'accuracy': accuracy, 'aiming': aiming } )
	model.load_weights( session_dir + '/checkpoints/ckpt' ).expect_partial()
	
	try :
		for image, target in ds_train :

			# Feed the image to the model:
			output = tf.squeeze( model( tf.expand_dims( image, 0 ) ) )
			#print( output.numpy() )

			#if output[0] < 0 :
				#continue

			# Display the image:
			image = tf.clip_by_value( ( image + 1 )/2, 0, 1 )
			plt.imshow( image )
			
			# Draw the reticles:
			for i in range( grid_w ) :
				for j in range( grid_h ) :
					if output[i,j,0] > 0.5 :
						x = ( output[i,j,1] + i )/grid_w*input_w
						y = ( output[i,j,2] + j )/grid_h*input_h
						reticle_color = 'r'
						plt.axvline( x, c=reticle_color )
						plt.axhline( y, c=reticle_color )
						if output[i,j,3] > 0 :
							r = output[i,j,3]*max( input_w, input_h )/2
							plt.gcf().gca().add_artist( plt.Circle( ( x, y ), r, fill=False, color=reticle_color ) )

			plt.show()

	except KeyboardInterrupt :
		pass
	print()
	exit( 0 )



############
# Training #
############

#ds_train = ds_train.shuffle( 1000 )
ds_train = ds_train.batch( 64 )
ds_train = ds_train.prefetch( tf.data.experimental.AUTOTUNE )

ds_val = ds_val.batch( 64 )
ds_val = ds_val.prefetch( tf.data.experimental.AUTOTUNE )


model.compile(
	optimizer=tf.keras.optimizers.SGD( learning_rate=0.001, momentum=0.9 ),
	#optimizer=tf.keras.optimizers.Adam( 0.001 ),
	loss=detection_loss,
	metrics=[ accuracy, aiming ]
)


ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
	#filepath=session_dir + '/checkpoints/ckpt_{epoch:03d}',
	filepath=session_dir + '/checkpoints/ckpt',
	save_weights_only=True,
	save_freq='epoch',
	#monitor='val_Fscore',
	#mode='max',
	#save_best_only=True
)

#monitor = Monitor( [ 2 ]*3, labels=[ 'training', 'validation' ], titles=[ 'Loss', 'Precision', 'Recall', 'Aiming distance' ] )

class stats_callback( tf.keras.callbacks.Callback ) :

	def __init__( self, log_path ) :
		self._log_path = log_path
		with open( self._log_path, 'a' ) as f :
			f.write( '1:epoch 2:loss 3:val_loss 4:accuracy 5:val_accuracy 6:aiming 7:val_aiming\n' )

	def on_epoch_end( self, epoch, logs=None ) :
		stats = ( epoch + 1, logs['loss'], logs['val_loss'],
		                     logs['accuracy'], logs['val_accuracy'],
		                     logs['aiming'], logs['val_aiming'] )
		#monitor.add_data( *stats )
		with open( self._log_path, 'a' ) as f :
			f.write( '%i %f %f %f %f %f %f\n' % stats )

try :
	model.fit(
		ds_train,
		epochs=2000,
		steps_per_epoch=100,
		validation_data=ds_val,
		validation_steps=3,
		callbacks=[ ckpt_callback, stats_callback( session_dir  + '/stats.log' ) ]
	)
except KeyboardInterrupt :
	pass


#model.save( session_dir + '/model' )
#print( 'Model saved in %s' % session_dir )
