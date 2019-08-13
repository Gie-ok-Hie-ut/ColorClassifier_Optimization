import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))


#####Extract
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    print("Here")
    print(i)
    model.set_input(data)

    model.extract_multi()
    #visuals = model.get_current_visuals_test()
    #img_path = model.get_image_paths()
    #print('%04d: process image... %s' % (i, img_path))
    #visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

######Centroid
model.calc_centroid_multi()
model.save_centroid_multi()
#model.load_centroid()


######New Calc_Distance
#for i, data in enumerate(dataset):
#    if i >= opt.how_many:
#        break
#    model.set_input(data)
#    model.calc_distance()

######New Calc_Transfer
#for i, data in enumerate(dataset):
#    if i >= opt.how_many:
#        break
#    model.set_input(data)
#    model.calc_transfer()
#    
#    visuals = model.get_current_visuals_transfer(i)
#    #img_path = model.get_image_paths()
#    img_path='A_paths'
#    print('%04d: process image... %s' % (i, img_path))
#    visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

webpage.save()
