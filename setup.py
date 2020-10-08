from setuptools import setup 

setup(name = "TagThis",
        author = "Yeou Chiou",
        author_email = "yeouchiou@gmail.com",
        url = "https://github.com/yeouchiou/TagThis",
        packages = ['TagThis'],
        package_data = {'TagThis': ['data/*']},
        include_package_data = True

)
