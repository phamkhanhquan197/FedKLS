"""Flower tested datasets information."""

dataset_info = { # Image datasets
                'cifar10': {
                            'num_classes': 10,
                            'output_column': 'label',
                            'test_set': 'test'
                            },
                'cifar100': {
                            'num_classes': 100,
                            'output_column': 'fine_label',
                            'test_set': 'test'
                            },
                'mnist': {
                            'num_classes': 10,
                            'output_column': 'label',
                            'test_set': 'test'
                            },
                'fashion_mnist': {
                            'num_classes': 10,
                            'output_column': 'label',
                            'test_set': 'test'
                            },
                'sasha/dog-food': {
                            'num_classes': 2,
                            'output_column': 'label',
                            'test_set': 'test'
                            },
                'zh-plus/tiny-imagenet': {
                            'num_classes': 200,
                            'output_column': 'label',
                            'test_set': 'valid'
                            },
                'Mike0307/MNIST-M': {
                            'num_classes': 10,
                            'output_column': 'label',
                            'test_set': 'test'
                            },
                'flwrlabs/usps': {
                            'num_classes': 10,
                            'output_column': 'label',
                            'test_set': 'test'
                            },
                }