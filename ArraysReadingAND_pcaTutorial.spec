# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['ArraysReadingAND_pcaTutorial.py'],
             pathex=['/Users/otheruser/Mirror/packaging_tutorial/MaterialsInformatics/MaterialsInformatics'],
             binaries=[],
             datas=[],
             hiddenimports=['tkinter.filedialog','tkinter','glob','sys','os','unsupervised_learning_tools','numpy','AutomationTools','MachineLearningTools','matplotlib','pylab','matplotlib.mlab','scipy.stats','re','sklearn.decomposition','PIL','cv2','sys','importlib','webcolors','extcolors'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=True)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [('v', None, 'OPTION')],
          exclude_binaries=True,
          name='ArraysReadingAND_pcaTutorial',
          debug=True,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='ArraysReadingAND_pcaTutorial')
