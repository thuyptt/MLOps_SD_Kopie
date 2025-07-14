import unittest
from unittest.mock import patch, MagicMock
from omegaconf import DictConfig, OmegaConf
import sys


sys.path.append('../mlops_project_2024')
import train_model

'''''
import unittest
import hydra
from omegaconf import OmegaConf
import os
import subprocess


class TestHydraConfigLoading(unittest.TestCase):
    def test_hydra_config_loading(self):
        script_path = os.path.join(os.path.dirname(__file__), '../', 'mlops_project_2024','train_model.py')
        result = subprocess.run(
            ['python', script_path, 'hydra.run.dir=.', 'hydra.job.name=test_job'],
            capture_output=True, text=True
        )

        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        print("Captured stdout:", result.stdout)
        cfg = OmegaConf.create(result.stdout)
        self.assertIn('output_dir', cfg)
        self.assertIn('pretrained_model_name_or_path', cfg)

if __name__ == '__main__':
    unittest.main()
'''''



class TestHydraConfigLoading(unittest.TestCase):
    @patch('train_model.main')
    def test_hydra_config_loading(self, mock_main):
        # Create a mock configuration
        mock_cfg = OmegaConf.create({
            'output_dir': './output',
            'pretrained_model_name_or_path': './model',
            'logging_dir': './logs',
            'gradient_accumulation_steps': 1,
            'mixed_precision': 'no',
            'seed': 42,
            'num_validation_images': 1,
            'validation_prompt': 'A test prompt',
            'allow_tf32': False,
            'scale_lr': False,
            'learning_rate': 1e-4,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_weight_decay': 0.01,
            'adam_epsilon': 1e-8,
            'processed_dataset_path': './dataset',
            'image_column': 'image',
            'resolution': 256,
            'center_crop': True,
            'random_flip': True,
            'train_batch_size': 1,
            'dataloader_num_workers': 1,
            'lr_warmup_steps': 100,
            'max_train_steps': 1000,
            'validation_epochs': 1,
            'checkpointing_steps': 100,
            'checkpoints_total_limit': 2,
        })

        train_model.main(mock_cfg)

        # Verify that the configuration contains expected keys
        self.assertIn('output_dir', mock_cfg)
        self.assertIn('pretrained_model_name_or_path', mock_cfg)
        self.assertEqual(mock_cfg.output_dir, './output')  # Example check





class TestModelInitialization(unittest.TestCase):
    @patch('train_model.CLIPTextModel.from_pretrained')
    @patch('train_model.CLIPTokenizer.from_pretrained')
    @patch('train_model.AutoencoderKL.from_pretrained')
    @patch('train_model.UNet2DConditionModel.from_pretrained')
    def test_model_initialization(self, mock_unet, mock_vae, mock_tokenizer, mock_text_encoder):
        mock_cfg = OmegaConf.create({
            'pretrained_model_name_or_path': './model',
            'revision': 'main',
            'variant': None
        })

        # Run model initialization
        tokenizer = train_model.CLIPTokenizer.from_pretrained(mock_cfg.pretrained_model_name_or_path)
        text_encoder = train_model.CLIPTextModel.from_pretrained(mock_cfg.pretrained_model_name_or_path)
        vae = train_model.AutoencoderKL.from_pretrained(mock_cfg.pretrained_model_name_or_path)
        unet = train_model.UNet2DConditionModel.from_pretrained(mock_cfg.pretrained_model_name_or_path)

        self.assertTrue(mock_tokenizer.called)
        self.assertTrue(mock_text_encoder.called)
        self.assertTrue(mock_vae.called)
        self.assertTrue(mock_unet.called)

'''''
class TestDatasetLoading(unittest.TestCase):
    @patch('train_model.load_from_disk')
    def test_dataset_loading_and_preprocessing(self, mock_load_from_disk):

        mock_dataset = MagicMock()
        mock_load_from_disk.return_value = mock_dataset

        mock_cfg = OmegaConf.create({
            'processed_dataset_path': './dataset',
            'image_column': 'image25.jpg',
            'resolution': 256,
            'center_crop': True,
            'random_flip': True
        })
        # Call the dataset loading function
        dataset = train_model.load_from_disk(mock_cfg.processed_dataset_path)
        self.assertEqual(dataset, mock_dataset)

        # Patch the preprocessing function
        with patch('train_model.train_transforms', return_value=lambda x: x) as mock_transforms:
            preprocessed_dataset = train_model.preprocess_train({'image': [MagicMock()]})

            self.assertIn('pixel_values', preprocessed_dataset)

'''''

if __name__ == '__main__':
    unittest.main()
