import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem import QED
from typing import Callable


def calculate_molecular_descriptors(smiles: str) -> pd.DataFrame:
    """
    Calculate molecular descriptors using RDKit.

    Args:
        smiles (str): A SMILES representation of a molecule.

    Returns:
        pandas.DataFrame: A DataFrame containing the calculated molecular descriptors.
    """
    # TRANSFORM MOLECULES SMILES TO RDKit MOL OBJECT
    molecule = Chem.MolFromSmiles(smiles)

    # MOLECULAR DESCRIPTORS CALCULATION OBJECT
    calculation = MoleculeDescriptors.MolecularDescriptorCalculator(
        [x[0] for x in Descriptors._descList]
    )
    molecular_descriptors_names = calculation.GetDescriptorNames()

    # ADD HIDROGENS TO MOLECULES
    molecule = Chem.AddHs(molecule)

    # CALCULATE MOLECULAR DESCRIPTORS
    molecular_descriptors = calculation.CalcDescriptors(molecule)

    # MQN NAMES
    mqn_names = ["MQN" + str(i) for i in range(1, 43)]

    # CALCULATE MQN DESCRIPTORS
    calculation_mqn = rdMolDescriptors.MQNs_(molecule)

    # JOIN MOLECULAR DESCRIPTORS AND MQN DESCRIPTORS
    molecular_descriptors = list(molecular_descriptors) + calculation_mqn

    # JOIN MOLECULAR DESCRIPTORS NAMES AND MQN NAMES
    molecular_descriptors_names = list(molecular_descriptors_names) + mqn_names

    return pd.DataFrame([molecular_descriptors], columns=molecular_descriptors_names)


def calculate_molecule_set(
    smiles_set: pd.DataFrame, smiles_column: str, function: Callable
) -> pd.DataFrame:
    """
    Calculate molecular descriptors for each molecule in a pandas DataFrame.

    Args:
        smiles_set (pandas.DataFrame): A DataFrame containing SMILES representations of
        molecules.
        smiles_column (str): The name of the column in the DataFrame containing the
        SMILES strings.
        function (function): A function that takes a SMILES string as input and returns
        a DataFrame of molecular descriptors.

    Returns:
        pandas.DataFrame: A DataFrame containing the input DataFrame with added
        columns for the calculated molecular descriptors.
    """
    # CALCULATE MOLECULAR DESCRIPTORS FOR EACH MOLECULE

    molecular_descriptors = smiles_set[smiles_column].apply(function)

    # CONCATENATE MOLECULAR DESCRIPTORS DATAFRAMES

    molecular_descriptors = pd.concat(molecular_descriptors.values).reset_index(
        drop=True
    )

    # CONCATENATE MOLECULES SMILES AND MOLECULAR DESCRIPTORS DATAFRAMES
    smiles_set = pd.concat([smiles_set, molecular_descriptors], axis=1)

    # RETURN MOLECULES SMILES AND MOLECULAR DESCRIPTORS DATAFRAME
    return smiles_set


def calculate_maccs_keys(smiles: str) -> pd.DataFrame:
    """
    Calculate MACCS keys for a molecule.

    Args:
        smiles (str): A SMILES representation of the molecule.

    Returns:
        pandas.DataFrame: A DataFrame containing the MACCS keys for the molecule.
    """
    # TRANSFORM MOLECULES SMILES TO RDKit MOL OBJECT
    molecule = Chem.MolFromSmiles(smiles)

    # DATAFRAM COLUMN NAMES
    maccs_keys_names = ["maccs_" + str(i) for i in range(167)]

    # CALCULATE MACCS KEYS
    maccs_keys = list(MACCSkeys.GenMACCSKeys(molecule).ToBitString())

    # RETURN MACCS KEYS
    return pd.DataFrame([maccs_keys], columns=maccs_keys_names)


def calculate_ecfp4_fingerprints(smiles: str) -> pd.DataFrame:
    """
    Calculate the ECPF4 fingerprints for a given molecule.

    Args:
        smiles (str): A SMILES string representing the input molecule.

    Returns:
        pandas.DataFrame: A DataFrame containing the ECPF4 fingerprints for the
        molecule. The DataFrame has one row and 1024 columns,corresponding to the
        1024 bits in the ECPF4 fingerprint. The column names are of the form "ecfp4X",
        where X is a number from 1 to 1024.
    """
    # TRANSFORM MOLECULES SMILES TO RDKit MOL OBJECT
    molecule = Chem.MolFromSmiles(smiles)

    # DATAFRAME COLUMN NAMES
    nBits = 1024
    ecfp4_fingerprints_names = ["ecfp4_" + str(i + 1) for i in range(nBits)]

    # CALCULATE ECPF4 KEYS
    ecfp4_fingerprints = list(
        AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=nBits)
    )

    # RETURN ECPF4 KEYS
    return pd.DataFrame([ecfp4_fingerprints], columns=ecfp4_fingerprints_names)


# def calculate_klekota_roth_keys(smiles: str) -> pd.DataFrame:
#     """
#     Calculate Klekota-Roth keys for a molecule.
#     """
#     # TRANSFORM MOLECULES SMILES TO RDKit MOL OBJECT
#     molecule = Chem.MolFromSmiles(smiles)

#     # CALCULATE KLEKOTA-ROTH KEYS
#     klekota_roth_keys = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
#         molecule, nBits=1024
#     )

#     # RETURN KLEKOTA-ROTH KEYS
#     return klekota_roth_keys


def calculate_qed_properties(smiles: str) -> pd.DataFrame:
    """
    Calculate quantitative estimate of drug-likeness (QED) properties for a molecule.

    Args:
    smiles (str): SMILES representation of the molecule.

    Returns:
    pandas.DataFrame: A dataframe containing QED properties for the molecule.

    Raises:
    RDKitError: If the SMILES string cannot be converted to an RDKit molecule object.
    """
    # TRANSFORM MOLECULES SMILES TO RDKit MOL OBJECT
    molecule = Chem.MolFromSmiles(smiles)

    # CALCULATE QED PROPERTIES
    qed_properties = QED.properties(molecule)

    # RETURN QED PROPERTIES
    return pd.DataFrame([qed_properties])
