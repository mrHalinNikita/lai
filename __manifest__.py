{
    'name': 'LAI Estimator for Chernozem Region',
    'version': '1.0',
    'summary': 'Calculate Leaf Area Index from RGB images for Central Black Soil Region',
    'category': 'Agriculture',
    'author': 'Halin Nikita',
    'depends': ['base', 'web', 'website'],
    'data': [
        'data/ir.model.access.csv',
        'views/templates.xml',
    ],
    'assets': {
        'web.assets_frontend': [
            # SCSS
            'lai_estimator/static/src/scss/lai_calculator.scss',
            # JS
            'lai_estimator/static/src/js/lai_calculator.js',
        ],
    },
    'external_dependencies': {
        'python': [
            'numpy',
            'opencv-python',
            'scikit-learn',
            'matplotlib',
            'Pillow',
        ]
    },
    'installable': True,
    'application': True,
    'auto_install': False,
    'license': 'LGPL-3',
}