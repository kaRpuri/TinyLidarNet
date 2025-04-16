from setuptools import setup

package_name = 'tinylidarnet'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'tensorflow',
        'matplotlib',
        'scikit-learn',
        # 'rosbag2_py',  # Only if available via pip, otherwise install via ROS2 tools
    ],
    zip_safe=True,
    description='tinylidarnet',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tinylidarnet = tinylidarnet.tinylidarnet:main',
            # Add other scripts as needed
        ],
    },
)
