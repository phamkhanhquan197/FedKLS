"""Flower tested datasets information."""

dataset_info = { # Image datasets
                'cifar10': {
                            'num_classes': 10,
                            'output_column': 'label',
                            'test_set': 'test',
                            'input_shape' : (3, 32, 32)
                            },
                'cifar100': {
                            'num_classes': 100,
                            'output_column': 'fine_label',
                            'test_set': 'test',
                            'input_shape' : (3, 32, 32)
                            },
                'mnist': {
                            'num_classes': 10,
                            'output_column': 'label',
                            'test_set': 'test',
                            'input_shape' : (1, 28, 28)
                            },
                'fashion_mnist': {
                            'num_classes': 10,
                            'output_column': 'label',
                            'test_set': 'test',
                            'input_shape' : (1, 28, 28)
                            },
                'sasha/dog-food': {
                            'num_classes': 2,
                            'output_column': 'label',
                            'test_set': 'test',
                            'input_shape' : (3, 225, 225)
                            },
                'zh-plus/tiny-imagenet': {
                            'num_classes': 200,
                            'output_column': 'label',
                            'test_set': 'valid',
                            'input_shape' : (3, 64, 64)
                            },
                'Mike0307/MNIST-M': {
                            'num_classes': 10,
                            'output_column': 'label',
                            'test_set': 'test',
                            'input_shape' : (3, 32, 32)
                            },
                'flwrlabs/usps': {
                            'num_classes': 10,
                            'output_column': 'label',
                            'test_set': 'test',
                            'input_shape' : (1, 16, 16)
                            },
                }