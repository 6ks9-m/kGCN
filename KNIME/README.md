# KNIME_GCN-K

KNIME node extension for GCN-K (GraphCNN).

## Requirements

GraphCNN  
Anaconda3 and python environment required by GraphCNN

## Anaconda���̃Z�b�g�A�b�v (�J���ҁA���[�U����)

�ŐV��Anaconda2018���g�����Ƃ���Apandas�̃C���|�[�g�Ŗ��ɂȂ����̂ŉߋ��̃o�[�W�������g��  
https://repo.continuum.io/archive/  
����  
Anaconda3-5.3.1-Linux-x86_64.sh(Windows�ł�Anaconda3-5.3.1-Windows-x86_64.exe)���_�E�����[�h���ăC���X�g�[���B  

�[��(anaconda prompt)��
```
conda update conda
conda create -n GraphCNN python=3.6 # �ŐV��3.7�ł�tensorflow�̃C���X�g�[�������܂������Ȃ�����
conda activate GraphCNN
conda install -c rdkit rdkit=2017.* # RDKit�͏����o�[�W������������K�v��������
python -m pip install --upgrade pip
pip install --ignore-installed --upgrade tensorflow
pip install joblib
pip install keras
pip install matplotlib
pip install seaborn
pip install IPython
pip install scikit-learn
```

�����s����  
```
ImportError: Something is wrong with the numpy installation. 
While importing we detected an older version of numpy in ['/home/furukawa/anaconda3/envs/GraphCNN/lib/python3.5/site-packages/numpy']. 
One method of fixing this is to repeatedly uninstall numpy until none is found, then reinstall this version.
```
�ƃG���[���o��ꍇ�́A����ꂽ�Ƃ���
```
pip uninstall numpy
pip uninstall numpy
pip uninstall numpy
pip install numpy
```
�Ƃ���B


## GraphCNN�̃Z�b�g�A�b�v (�J���ҁA���[�U����)

GraphCNN��github����_�E�����[�h���ēW�J(�܂���git clone)

```
pip install -e GraphCNN/gcnvisualizer
pip install -r GraphCNN/gcnvisualizer/requirements.txt
pip install bioplot
```

�ȉ��A����m�F  
���ϐ���GCNK_SOURCE_PATH��GraphCNN�̃p�X��ݒ�  
���ϐ���GCNK_PYTHON_PATH��python.sh(Windows�ł�python.bat)�̃p�X��ݒ�(Anaconda���z�����g���Ă��邽��activate���K�v�ɂȂ�B�����łȂ��ꍇ��python���Z�b�g�����OK)  
���ϐ���PYTHONPATH��GCNK_SOURCE_PATH��ǉ�  
testdata/singletask�t�H���_��run.sh(Windows�ł�run.bat)�����s���ē���m�F  

## �J�����̃Z�b�g�A�b�v (�J���҂̂�)

V3.6����SDK�̔z�z�͂��Ă��Ȃ��̂ŁA�Z�b�g�A�b�v�͏����ʓ|�B  
v3.5�̂��̂��g���B

https://www.knime.com/download-previous-versions  
����KNIME SDK version 3.5.3 for Windows���_�E�����[�h���ăC���X�g�[��

KNIME SDK���N��  
Workspace�ɃN���[���������|�W�g���̃t�H���_���w��(�ȉ��AC:\work\KNIME_GCN-K �Ƃ���)

[File]-[Open Projects from File Ssytem...]  
Import source �� C:\work\KNIME_GCN-K\GCN-K ��I��

## �f�o�b�O���s (�J���҂̂�)

GCN-K�v���W�F�N�g���E�N���b�N��[Run As]-[Eclipse Application]  
[Window]-[Perspective]-[Open Perspective]-[Other]  
KNIME��I������OK  
Node Repository�ɍ쐬�����m�[�h���ǉ�����A�g�p�ł���

## �m�[�h���W���[���̍쐬 (�J���҂̂�)

GCN-K�v���W�F�N�g���E�N���b�N��[Export]  
[Plug-in Development]-[Deployable plug-ins and fragments]��I������Next  
�K����Directory��I������Finish  
�I������Directory��plugins�ȉ���jar�t�@�C�������������

## �m�[�h���W���[����KNIME�ɃC���X�g�[�� (���[�U�̂�)
KNIME���C���X�g�[��  
https://www.knime.com/downloads  
jar�t�@�C����KNIME��dropins�f�B���N�g��(Windows�ł͒ʏ�C:\Program Files\KNIME\dropins)�ɃR�s�[����  

## �e�X�g(���[�U�̂�)
���L���Q��  
[�V���O���^�X�N](testdata/singletask/README.md)  
[�}���`�^�X�N](testdata/multitask/README.md)  
[�}���`���[�_��(����)](testdata/multimodal/README.md)  

