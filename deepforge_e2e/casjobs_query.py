import os

from SciServer.Authentication import login
from SciServer.Config import isSciServerComputeEnvironment
from SciServer.CasJobs import executeQuery


CASJOBS_QUERY = """
SELECT TOP 10000 za.specObjID, za.bestObjID, za.class, za.subClass, za.z, za.zErr,
  po.objID, po.type, po.flags, po.ra, po.dec, po.run, po.rerun, po.field, po.camcol,
  (po.petroMag_r-po.extinction_r) as dered_petro_r,
  zp.z as zphot, zp.zErr as dzphot,
  zi.e_bv_sfd,zi.primtarget, zi.sectarget,zi.targettype,zi.spectrotype,zi.subclass

INTO MyDB.{}
FROM SpecObjAll za
  JOIN PhotoObjAll po ON (po.objID = za.bestObjID)
  JOIN Photoz zp ON (zp.objID = za.bestObjID)
  JOIN galSpecInfo zi ON (zi.SpecObjID = za.specObjID)
WHERE
  (za.z>0 AND za.zWarning=0)
    AND (za.targetType='SCIENCE' AND za.survey='sdss')
    AND (za.class='GALAXY' AND zi.primtarget>=64 AND zi.targettype='GALAXY')
    AND (po.clean=1 AND po.insideMask=0)
  AND ((po.petroMag_r-po.extinction_r)<=17.8)
  AND za.z <= 0.4
"""


class CASJOBSQuery():
    """Perform a custom casjobs query in the SciServer-CASJOBs
    Parameters
    ----------
    casjobs_query: (str), the query to run in CASJOBS Server
    context: The context to run the query (DR12, MYDB, default: "DR12")

    Notes:
    ------
    While running this, please provide your username and password as environment variables
                        SCISERVER_USERNAME, SCISERVER_PASSWORD
    """
    def __init__(self, casjobs_query=None, context="DR12"):
        self.context = context
        if not casjobs_query:
            self.casjobs_query = CASJOBS_QUERY.format("SDSS_" + self.context)
        self.table_name = "SDSS_" + self.context
        self._logged_in = False

    def execute(self):
        """Execute the query in CASJOBS"""
        if not self._logged_in:
            token = self._login()
        self.drop_table()
        executeQuery(self.casjobs_query, context=self.context, format='pandas')
        query_df = executeQuery("SELECT * FROM MYDB.{}".format(self.table_name), format='pandas')
        return query_df

    def _login(self):
        """If not logged in and not inside sciserver login"""
        if isSciServerComputeEnvironment():
            self._logged_in = True
            return
        try:
            username = os.environ['SCISERVER_USERNAME']
            password = os.environ['SCISERVER_PASSWORD']
        except KeyError:
            self._logged_in = False
            raise Exception('Username and password not found')

        token = login(username, password)
        self._logged_in = True
        return token

    def drop_table(self):
        drop_table_query = "DROP TABLE IF EXISTS {}".format(self.table_name)
        if not self._logged_in:
            self._login()
        executeQuery(drop_table_query, context='MyDB')

