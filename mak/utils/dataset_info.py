"""Flower tested datasets information."""

dataset_info = { # Image datasets
                'cifar10': {
                            'num_classes': 10,
                            'output_column': 'label',
                            'test_column': 'test'
                            },
                'cifar100': {
                            'num_classes': 100,
                            'output_column': 'fine_label',
                            'test_column': 'test'
                            },
                'mnist': {
                            'num_classes': 10,
                            'output_column': 'label',
                            'test_column': 'test'
                            },
                'fashion_mnist': {
                            'num_classes': 10,
                            'output_column': 'label',
                            'test_column': 'test'
                            },
                'sasha/dog-food': {
                            'num_classes': 2,
                            'output_column': 'label',
                            'test_column': 'test'
                            },
                'zh-plus/tiny-imagenet': {
                            'num_classes': 200,
                            'output_column': 'label',
                            'test_column': 'valid'
                            },
                'Mike0307/MNIST-M': {
                            'num_classes': 10,
                            'output_column': 'label',
                            'test_column': 'test'
                            },
                'flwrlabs/usps': {
                            'num_classes': 10,
                            'output_column': 'label',
                            'test_column': 'test'
                            },
                }