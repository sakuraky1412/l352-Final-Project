import os
import sys
import cv2
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import sewar.full_ref as fr
from proto.eval_config_pb2 import EvalConfig
from google.protobuf import text_format
from core import evaluator as evaluator
from core import evaluator_interactive as evaluator_interactive

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('config_file', './model/eval.config', 'Path of config file')
flags.DEFINE_integer('id', 0, 'ID of config file')
FLAGS = flags.FLAGS
id = FLAGS.id

os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % (id % 4)


def get_configs():
    eval_config = EvalConfig()
    with open(FLAGS.config_file, 'r') as f:
        # parses a text representation of config_file into eval_config
        text_format.Merge(f.read(), eval_config)
    tf.logging.info(eval_config)
    return eval_config


# credit to https://stackoverflow.com/questions/42594993/gradient-mask-blending-in-opencv-python
def alphaBlend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0

    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)

    return blended

def improve_dof(eval_config):
    img_path = eval_config.image_path
    res_path = eval_config.res_path + 'dof/'
    res_depth_path = eval_config.res_path + 'depth/'
    res_impr_path = eval_config.res_path + 'impr/'
    if not os.path.isdir(res_impr_path):
        os.mkdir(res_impr_path)
    image_names = os.listdir(img_path)
    for img_id, image_name in enumerate(image_names):
        if not image_name.startswith('.'):
            print("Processing Img: " + image_name + "\t%d/%d" % (img_id, len(image_names)))

            original_image_path = img_path + image_name
            image_path = res_depth_path + image_name
            dof_image_path = res_path + image_name

            image = cv2.imread(image_path)
            height, width, channels = image.shape
            print("Img Size:\t%d*%d" % (height, width))
            original_image = cv2.imread(original_image_path)
            dof_image = cv2.imread(dof_image_path)

            new_original_image = cv2.resize(original_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
            new_dof_image = cv2.resize(dof_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            start_time = time.time()
            lower = np.array([0, 70, 50], dtype="uint8")
            upper = np.array([10, 255, 255], dtype="uint8")
            mask1 = cv2.inRange(image, lower, upper)
            lower = np.array([170, 70, 50], dtype="uint8")
            upper = np.array([180, 255, 255], dtype="uint8")
            mask2 = cv2.inRange(image, lower, upper)
            mask = mask1 | mask2

            mask = cv2.GaussianBlur(mask, (11, 11), 1)
            blended = alphaBlend(new_original_image, new_dof_image, mask)
            blended = cv2.resize(blended, (dof_image.shape[1], dof_image.shape[0]),
                                 interpolation=cv2.INTER_AREA)
            cv2.imwrite(res_impr_path + image_name, blended)
            end_time = time.time()
            print('Spend time:\t%fs' % (end_time - start_time))

# method to produce psnr/ssim scores
def metric():
    eval_config = get_configs()
    img_path = eval_config.image_path
    res_path = eval_config.res_path + 'dof/'
    res_impr_path = eval_config.res_path + 'impr/'
    image_names = os.listdir(img_path)
    img_ids = []
    psnrs1 = []
    psnrs2 = []
    ssims1 = []
    ssims2 = []
    for img_id, image_name in enumerate(image_names):
        if not image_name.startswith('.'):
            img_ids.append(img_id)
            # Read images from file.
            im1 = cv2.imread(img_path + image_name)
            im2 = cv2.imread(res_path + image_name)
            im3 = cv2.imread(res_impr_path + image_name)
            im1 = cv2.resize(im1, (im2.shape[1], im2.shape[0]), interpolation=cv2.INTER_AREA)
            # Compute PSNR over tf.uint8 Tensors.
            psnr1 = fr.psnr(im1, im2, MAX=255)
            psnr2 = fr.psnr(im1, im3, MAX=255)
            psnrs1.append(psnr1)
            psnrs2.append(psnr2)

            # Compute SSIM over tf.uint8 Tensors.
            ssim1, cs = fr.ssim(im1, im2, MAX=255)
            ssim2, cs = fr.ssim(im1, im3, MAX=255)
            ssims1.append(ssim1)
            ssims2.append(ssim2)
    print(psnrs1)
    print(psnrs2)
    print(ssims1)
    print(ssims2)
    plt.plot(img_ids, psnrs1, label='original')
    plt.plot(img_ids, psnrs2, label='project')
    plt.legend(title='implementation')
    plt.xlabel('image id')
    plt.ylabel('psnr')
    plt.show()
    plt.plot(img_ids, ssims1, label='original')
    plt.plot(img_ids, ssims2, label='project')
    plt.legend(title='implementation')
    plt.xlabel('image id')
    plt.ylabel('ssim')
    plt.show()

def main(interactive="True"):
    eval_config = get_configs()
    if interactive == "True":
        evaluator_interactive.evaluate(eval_config)
    elif interactive == "False":
        evaluator.evaluate(eval_config)
    else:
        improve_dof(eval_config)
        # initial work
        # image_path = './data/res/depth/7100492477_234552dc97_b.jpg'
        # original_image_path = './data/imgs/7100492477_234552dc97_b.jpg'
        # dof_image_path = './data/res/dof/Figure_1.png'
        # image = cv2.imread(image_path)
        # original_image = cv2.imread(original_image_path)
        # dof_image = cv2.imread(dof_image_path)
        #
        # new_original_image = cv2.resize(original_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        # new_dof_image = cv2.resize(dof_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #
        # lower = np.array([0, 70, 50], dtype="uint8")
        # upper = np.array([10, 255, 255], dtype="uint8")
        # mask1 = cv2.inRange(image, lower, upper)
        # lower = np.array([170, 70, 50], dtype="uint8")
        # upper = np.array([180, 255, 255], dtype="uint8")
        # mask2 = cv2.inRange(image, lower, upper)
        # mask = mask1 | mask2
        #
        # mask = cv2.GaussianBlur(mask, (11, 11), 1)
        # blended = alphaBlend(new_original_image, new_dof_image, 255-mask)
        # blended = cv2.resize(blended, (dof_image.shape[1], dof_image.shape[0]),
        #                            interpolation=cv2.INTER_AREA)
        # cv2.imwrite('./data/res/impr/Figure_1a.jpg', blended)
        # cv2.imshow('mask', mask)
        # cv2.imshow('new', new_dof_image)
        # cv2.imshow('original', new_original_image)
        # cv2.imshow('image', image)
        # cv2.imshow('blended', blended)
        # cv2.waitKey()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        main()
    else:
        interactive = sys.argv[1]
        main(interactive)

    # Plotting performance of deeplens
    # x = [213 * 320, 320 * 206, 213 * 320, 385 * 289, 386 * 289, 386 * 289, 339 * 578, 423 * 639, 388 * 577, 386 * 288, 387 * 288, 320 * 240, 387 * 288, 386 * 289, 387 * 287, 386 * 288, 384 * 512, 320 * 213, 386 * 288, 211 * 320, 388 * 578, 240 * 320, 240 * 320, 213 * 320, 385 * 287, 386 * 289, 320 * 180, 213 * 320, 385 * 578, 435 * 288, 320 * 240, 320 * 240, 434 * 578, 383 * 579, 385 * 288, 387 * 288, 435 * 572, 212 * 320, 384 * 580, 432 * 577, 387 * 288, 320 * 213, 388 * 289, 386 * 289, 387 * 288, 320 * 240, 320 * 275, 384 * 289, 386 * 287, 385 * 287, 387 * 290]
    # y = [0.044286, 0.070872, 0.030096, 0.080669, 0.063119, 0.047435, 0.044665, 0.057041, 0.033268, 0.053087, 0.058835, 0.048440, 0.058138, 0.044680, 0.041150, 0.057469, 0.043511, 0.037656, 0.054935, 0.036751, 0.039443, 0.044161, 0.043377, 0.030800, 0.054688, 0.063044, 0.040523, 0.040205, 0.039573, 0.058028, 0.041963, 0.039071, 0.046046, 0.036331, 0.050553, 0.050110, 0.037062, 0.052462, 0.039406, 0.042568, 0.058108, 0.038931, 0.057778, 0.047027, 0.064182, 0.035715, 0.050457, 0.058924, 0.056861, 0.043543, 0.040285]
    # x1 = [2054 * 3081, 7012 * 4535, 3745 * 5617, 771 * 579, 773 * 578, 772 * 578, 339 * 578, 3264 * 4928, 388 * 577, 773 * 576, 774 * 577, 2048 * 1536, 775 * 577, 772 * 578, 774 * 575, 772 * 577, 768 * 1024, 5184 * 3456, 773 * 577, 1356 * 2048, 388 * 578, 1536 * 2048, 3240 * 4320, 3183 * 4774, 771 * 575, 773 * 579, 4608 * 2592, 1365 * 2048, 385 * 578, 870 * 577, 2978 * 2234, 4320 * 3240, 434 * 578, 383 * 579, 770 * 577, 774 * 577, 435 * 572, 1554 * 2335, 384 * 580, 432 * 577, 775 * 576, 5472 * 3648, 776 * 579, 773 * 578, 774 * 577, 2048 * 1536, 3508 * 3024, 769 * 578, 773 * 575, 771 * 574, 775 * 580]
    # x2 = [2054 * 3081, 7012 * 4535, 3745 * 5617, 771 * 579, 773 * 578, 772 * 578, 339 * 578, 3264 * 4928, 388 * 577,
    #      773 * 576, 774 * 577, 2048 * 1536, 775 * 577, 772 * 578, 774 * 575, 772 * 577, 768 * 1024, 5184 * 3456,
    #      773 * 577, 1356 * 2048, 388 * 578, 1536 * 2048, 3240 * 4320, 3183 * 4774, 771 * 575, 773 * 579, 4608 * 2592,
    #      1365 * 2048, 385 * 578, 870 * 577, 2978 * 2234, 4320 * 3240, 434 * 578, 383 * 579, 770 * 577, 774 * 577,
    #      435 * 572, 1554 * 2335, 384 * 580, 432 * 577, 775 * 576, 5472 * 3648, 776 * 579, 773 * 578, 774 * 577,
    #      2048 * 1536, 3508 * 3024, 769 * 578, 773 * 575, 771 * 574, 775 * 580]
    #
    # y1 = [25.336139, 16.140840, 21.600682, 7.375443, 6.612470, 6.665198, 5.315901, 25.999394, 4.741604, 6.603207, 6.771327, 25.868804, 6.633869, 6.592156, 6.529518, 7.112547, 15.435083, 19.525409, 6.607416, 17.165610, 4.710007, 24.197184, 21.803112, 21.547754, 8.274998, 6.672580, 19.593406, 25.382696, 5.644108, 7.515303, 31.126371, 30.742581, 5.244219, 4.755229, 6.856987, 6.728703, 5.234297, 17.707861, 4.734025, 5.155862, 6.594329, 19.453825, 6.666151, 7.347489, 6.688542, 28.928624, 30.402713, 6.678897, 6.697050, 6.829008, 6.718377]
    # y2 = [31.013813, 19.698171, 24.187435, 8.300124, 7.407509, 7.845652, 5.411160, 29.586150, 5.643163, 7.948113, 11.138058, 32.913454, 9.256386, 7.316679, 6.264859, 6.253985, 13.551825, 38.308595, 11.254647, 34.413750, 5.879190, 36.615521, 55.723860, 44.440763, 12.048891, 9.887846, 20.777399, 24.225832, 6.680845, 10.171575, 39.058956, 40.389219, 6.969184, 6.859578, 7.678072, 7.365735, 5.723984, 23.883757, 5.870192, 5.582624, 8.233246, 30.053013, 8.250600, 6.754459, 6.546925, 43.372855, 56.340667, 9.637546, 7.602440, 10.125545, 10.949082]
    # x1, y1 = zip(*sorted(zip(x1, y1)))
    # x2, y2 = zip(*sorted(zip(x2, y2)))
    # x, y = zip(*sorted(zip(x, y)))
    # plt.plot(x1, y1, label='(0, 0))')
    # plt.plot(x2, y2, label='(160, 160))')
    # plt.legend(title='focus point:')
    # # plt.plot(x, y, label='project implementation')
    # plt.xlabel('image size')
    # plt.ylabel('execution time')
    # plt.show()
    # metric()
    # x1 = [23.197472844403038, 25.584866869961512, 25.89588445113637, 20.049258176224054, 24.78761447972772,
    #  21.409895306450935, 28.29625030591876, 32.03454190843131, 26.304639771472978, 20.08008616991408,
    #  24.109615932495284, 22.193356646624125, 18.740878089301802, 30.382299351867623, 20.24285752695151,
    #  20.75243418034209, 23.26468278822479, 20.27047321780038, 26.149212201378504, 27.344552393689085, 21.39101280150114,
    #  20.050397488439145, 24.286522721534038, 19.12234179361941, 21.56559400575034, 27.60457429310819,
    #  21.092963265423435, 25.38342931074765, 20.16897730119564, 20.666733565462753, 23.50439944922111,
    #  30.450476671073186, 19.457330373745435, 23.36660564262717, 24.670412114557614, 31.270120463394925,
    #  23.92819397327337, 24.601131728105536, 22.472237234315003, 21.029491834548818, 19.599503560139514,
    #  21.019045392658544, 19.460503031085423, 21.12022161663971, 21.077744560532274, 23.43328950509985,
    #  22.165836233665203, 23.53509640664816, 23.581526185287643, 24.914357513234236]
    # x2 = [25.018475860744076, 27.951199881894606, 27.775423222199286, 24.12935734859444, 28.84753207559889,
    #  25.834658422276306, 35.27453226087656, 32.42933570658473, 26.744923553361453, 23.456705951538105,
    #  28.90006717738934, 22.686230346733623, 24.391415537087692, 30.990710471934673, 22.615402209959537,
    #  22.923651627724265, 25.027871426559805, 23.875556560630834, 28.75672510111802, 28.945745913098804,
    #  21.591865244388938, 22.229442044543223, 25.72215646531413, 21.69774624055139, 23.28219636211849, 28.95961270660422,
    #  23.11734851550849, 29.409761694286082, 22.209201285852824, 21.21147894350247, 24.805679790544733,
    #  30.360658051712058, 30.121182304520403, 25.550709510166, 27.237328952612497, 33.118159918513456,
    #  25.855286458986285, 25.349715144233347, 23.322986134976226, 23.89798543185649, 22.77018670539767,
    #  26.570949808818497, 21.403870234364867, 25.20819939966868, 22.028637799167328, 26.50017667852147,
    #  23.811178111430706, 25.91181920976036, 24.703362343278567, 27.12418990466749]
    # y1 = [0.7137582605789379, 0.8282210464124221, 0.8637702929736061, 0.6315855432230749, 0.751913999048575,
    #  0.6846564549482516, 0.9194508519880475, 0.9047640795249157, 0.9372056502287439, 0.7538316297713585,
    #  0.7433591046969604, 0.7327721546583058, 0.5036997263396582, 0.8869449225887172, 0.6961716507393493,
    #  0.6123248471926335, 0.6914347392211182, 0.5962484697214333, 0.8064552138385827, 0.9260025568984837,
    #  0.7231154825929317, 0.6579111469493646, 0.8362988475372043, 0.6013439478582897, 0.6795119517812102,
    #  0.7680458787564876, 0.6173577808681814, 0.8378435457922112, 0.6592375638880145, 0.5275602775745029,
    #  0.6460817062710352, 0.955560669338694, 0.7238690347373803, 0.7552099488873857, 0.6920803405200259,
    #  0.8893587732859259, 0.7125169211532806, 0.8692006184132706, 0.8385509946663836, 0.6836664068023671,
    #  0.6459749493256312, 0.6625380668771349, 0.6478633015330043, 0.7475043189533362, 0.6748428004744959,
    #  0.8043994086352315, 0.7383476100985696, 0.6510368420769831, 0.6958333095157446, 0.8059200329574394]
    # y2 = [0.8122109294401155, 0.8658722621182738, 0.8756187385752519, 0.8444450646463708, 0.8895926366268375,
    #  0.856451731867959, 0.9647454966994822, 0.9115219463291117, 0.941555047356509, 0.8510090550455699,
    #  0.848797312975866, 0.7494034926359303, 0.862063700620018, 0.9013381264491122, 0.8072613096359031,
    #  0.7481725621940312, 0.7809966169478861, 0.8510849913837165, 0.8604812328495622, 0.9439240765414106,
    #  0.6844972027828211, 0.7317091721814116, 0.8596087368258449, 0.791339348510666, 0.8038671887628208,
    #  0.7936328112725523, 0.6789996025566385, 0.940140443156961, 0.7828968126651114, 0.5334676399158619,
    #  0.6931729644341474, 0.9526961349103936, 0.9821327112269747, 0.8021002059647738, 0.8260233436491723,
    #  0.9407514909418143, 0.809709385392058, 0.8921088705481307, 0.878773877389408, 0.850738572786022,
    #  0.7524113279209287, 0.8915851110499954, 0.7979490662084125, 0.8925666399542918, 0.6899380024820204,
    #  0.8083466336512878, 0.8270331600921318, 0.7576540569439422, 0.8098830402131464, 0.8630049871538125]
    #
    # cx1 = 0
    # tx1 = len(x1)
    # cy1 = 0
    # ty1 = len(y1)
    # print(len(x1))
    # for i in range(len(x1)):
    #     if x1[i] < x2[i]:
    #         cx1 += 1
    #     elif x1[i] == x2[i]:
    #         tx1 -= 1
    #
    #     if y1[i] < y2[i]:
    #         cy1 += 1
    #     elif y1[i] == y2[i]:
    #         ty1 -= 1
    #
    # print("%d/%d"%(cx1,tx1))
    # print("%d/%d"%(cy1,ty1))