
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'colorclassify': # FIVE K
        assert(opt.dataset_mode == 'fivek' or 'fivek2')
        from .colorclassify_model import ColorClassify_Model
        model = ColorClassify_Model()
    elif opt.model == 'colorclassify2': # AADB
        assert(opt.dataset_mode == 'aadb' or 'ava')
        from .colorclassify_model2 import ColorClassify_Model2
        model = ColorClassify_Model2()
    elif opt.model == 'colorclassify3': # AADB
        assert(opt.dataset_mode == 'ava')
        from .colorclassify_model3 import ColorClassify_Model3
        model = ColorClassify_Model3()
    elif opt.model == 'colorenhance':
        from .colorenhance_model import ColorEnhance_Model
        model = ColorEnhance_Model()
    elif opt.model == 'colorize':
        assert(opt.dataset_mode == 'aligned_seg')
        from .colorize_model import ColorizeModel
        model = ColorizeModel()
    elif opt.model == 'colorize_fcycle':
        assert(opt.dataset_mode == 'aligned_seg')
        from .colorize_fcycle_model import Colorize_fcycle_Model
        model = Colorize_fcycle_Model()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
