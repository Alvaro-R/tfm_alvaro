# IMPORT LIBRARIES
from chembl_webresource_client.new_client import new_client
import pandas as pd


def get_target_id_maximum_activity(
    organism_name: str, activity_type: str
) -> pd.DataFrame:
    """
    Get the target id with maximum number of molecules with activity IC50.

    Args:
        organism_name: Name of the organism.
        activity_type: Type of activity.

    Returns:
        The target ID with the maximum number of molecules with activity IC50.
    """
    # GET TARGETS FOR THAT ORGANISM
    target_client = new_client.target
    targets_id = target_client.filter(organism=organism_name).only(["target_chembl_id"])
    # TARGETS IDs TO LIST
    targets_id = [target["target_chembl_id"] for target in targets_id]

    # ACTIVITY CONNECTION
    activity_client = new_client.activity

    # GET THE MOLECULES FOR THAT TARGET WITH ACTIVITY IC50
    molecules = activity_client.filter(
        target_chembl_id__in=targets_id, standard_type=activity_type
    ).only(["target_chembl_id", "molecule_chembl_id"])
    # MOLECULES TO DATAFRAME
    molecules = pd.DataFrame.from_dict(molecules)
    # NUMBER OF MOLECULES FOR EACH TARGET
    molecules_counts = molecules.groupby("target_chembl_id").size()
    # TARGET ID WITH MAXIMUM NUMBER OF MOLECULES WITH ACTIVITY IC50
    target_id = molecules_counts.idxmax()
    return target_id


def get_molecules_from_target_activity(target_id: str, activity: str) -> pd.DataFrame:
    """
    Get the molecules for a target with a given activity.

    Args:
        target_id (str): Target ID.
        activity (str): Type of activity.

    Returns:
        pd.DataFrame: Molecules for a target with a given activity.
    """
    # ACTIVITY CONNECTION
    activity_client = new_client.activity
    # GET MOLECULES FROM TARGET
    molecules = activity_client.filter(
        target_chembl_id=target_id, standard_type=activity
    )
    return pd.DataFrame.from_dict(molecules)
