#...Imports...
#*************
import os
import numpy as np
import pickle
#
data_folder = 'data'
#
#...Functions...
#***************
def get_QT(smol, iso, T ):

  #...Read molecular data...
  #*************************
  print('lets get started')
  with open(os.path.join(data_folder, 'molecular_data.QTmol'), 'rb') as handle:
    MOLECULES = pickle.loads(handle.read())

  #...Get list molecule and molecular id...
  dict_molid = {}
  for key, val in MOLECULES.items():
    dict_molid[val['mol_id']] = key
 
  

  #...Get molecule name...
  #***********************

  sline = ''
  i = 0
  for key, val in dict_molid.items():
    i += 1
    sline += '{:2d} = {:<7s}'.format(key, val)
    if i % 5 == 0:
      sline += '\n'
  
  mol_data = MOLECULES[smol]
 
  
    #...Read QT file...
    #******************
  
  filename = os.path.join(data_folder, '{:d}.QTpy'.format(mol_data['mol_id']))
  print(filename)
  with open(filename, 'rb') as handle:
    QTdict = pickle.loads(handle.read())
  
  
    #...Compute QT...
    #****************
  
  T1 = np.floor(T)
  T2 = np.ceil(T)

  Q1 = QTdict[iso][T1]
  Q2 = QTdict[iso][T2]
  
  QT = Q1 + (Q2-Q1)*(T-T1)
  
#    sline = ' \n {:s} molecule      {:s} isotopologue '.format(smol, mol_data[iso]['tag'])
#    print(sline)
#    
#    sline = ' Q({:.3f}K) = {:f}  g(state independent weight) = {:s}'.format( T, QT, mol_data[iso]['g_si'])
#    print(sline)
    
  
  return QT, mol_data[iso]['g_si'],mol_data[iso]['tag']
  
##*************************************************************************************************


def is_float(sval):
  try:
    t = float(sval)
    return True
  except:
    return False
##*************************************************************************************************


# your code goes here: example to return Q(T) for H2O 161 at 296 K and CO2 636 at 200K

#...Functions...
#***************
def main( ):

  #...Read molecular data...
  #*************************
  print('lets get started')
  with open(os.path.join(data_folder, 'molecular_data.QTmol'), 'rb') as handle:
    MOLECULES = pickle.loads(handle.read())

  #...Get list molecule and molecular id...
  dict_molid = {}
  for key, val in MOLECULES.items():
    dict_molid[val['mol_id']] = key
 
  
  a=5
  while a==5:
    #...Get molecule name...
    #***********************

    sline = ''
    i = 0
    for key, val in dict_molid.items():
      i += 1
      sline += '{:2d} = {:<7s}'.format(key, val)
      if i % 5 == 0:
        sline += '\n'
    
    molflag = False
    while molflag is False:
      print(sline)
      usr_input = input('Enter molecule name or number (carriage return or 0 to stop): ')
      if len(usr_input) == 0:
        exit()
      if usr_input.isdecimal():
        if int(usr_input) == 0:
          exit()
        if int(usr_input) in dict_molid.keys():
          smol = dict_molid[int(usr_input)]
          molflag = True
          continue
        else:
          print('Molecular id not in range, try again')
          molflag = False
          continue
      else:
        if usr_input in MOLECULES.keys():
          smol = usr_input.strip()
          molflag = True
          continue
        else:
          print('Molecular name not in correct, try again')
          molflag = False
          continue
    
    mol_data = MOLECULES[smol]
 
  
    #...Get isotopologue...
    #**********************
  
    sline = ''
    i = 0
    for iso in mol_data['list_iso']:
      i += 1
      sline += '{:s} = {:<7s}'.format(iso, mol_data[iso]['tag'])
      if i % 5 == 0:
        sline += '\n'
  
    isoflag = False
    while isoflag is False:
      print(sline)
      usr_input = input('Enter isotopologue number: ')
      if usr_input not in mol_data['list_iso']:
        print('Isotopologue not in list, try again')
        isoflag = False
        continue
      else:
        iso = usr_input.strip()
        isoflag = True
        continue
        
    #...Read QT file...
    #******************
  
    filename = os.path.join(data_folder, '{:d}.QTpy'.format(mol_data['mol_id']))
    with open(filename, 'rb') as handle:
      QTdict = pickle.loads(handle.read())
  
  
   #...Get temperature...
    #*********************
  
    Tflag = False
    while Tflag is False:
      usr_input = input('Enter a temperature (1 to {:.1f} K): '.format(mol_data[iso]['Tmax']))
      if is_float(usr_input):
        T = float(usr_input)
        Tflag = True
        if T < 1 or T > mol_data[iso]['Tmax']:
          print('Entered temperature is not in range, try again')
          Tflag = False
          continue
      else:
        print('Entered temperature is not a float, try again')
        Tflag = False
        continue
  

    QT,gsi,iso_code = get_QT(smol, iso, T )
#    print('QT = ', QT)
    
    sline = ' \n {:s} molecule      {:s} isotopologue '.format(smol, mol_data[iso]['tag'])
    print(sline)
  
    
    sline = ' Q({:.3f}K) = {:f}  g(state independent weight) = {:s}'.format( T, QT, mol_data[iso]['g_si'])
    print(sline)
      
  
    usr_input = input('\n press carriage to continue; any other character to stop): ')
    if len(usr_input) == 0:
      a = 5
    else:
      exit()

##*************************************************************************************************
##************************************************************************************************
if __name__ == '__main__':

  #...Run main...
  main( )



