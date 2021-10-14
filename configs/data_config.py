from configs.paths_config import dataset_paths
import torchvision.transforms as transforms

train_img_transformer = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DATASETS = {
    'train_with_ffhq': {
        'transform': train_img_transformer,
        'train_images_root': dataset_paths['ffhq_images'],  # 使用95%的作为训练，5%的作为验证
        'test_images_root': dataset_paths['lfw-a']
    }
}











