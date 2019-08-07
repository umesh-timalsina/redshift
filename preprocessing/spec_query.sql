-- This query was run on the casJobs Server, and CSV File was download with the galaxies information
-- The authors of the paper used this query for their. ON DR12 it returned 547199 rows

SELECT za.specObjID, za.bestObjID, za.class, za.subClass, za.z, za.zErr,
  po.objID, po.type, po.flags, po.ra, po.dec,
  (po.petroMag_r-po.extinction_r) as dered_petro_r,
  zp.z as zphot, zp.zErr as dzphot,
  zi.e_bv_sfd,zi.primtarget, zi.sectarget,zi.targettype,zi.spectrotype,zi.subclass

INTO MyDB.SDSS_DR12
FROM SpecObjAll za
  JOIN PhotoObjAll po ON (po.objID = za.bestObjID)
  JOIN Photoz zp ON (zp.objID = za.bestObjID)
  JOIN galSpecInfo zi ON (zi.SpecObjID = za.specObjID)
WHERE
  (za.z>0 AND za.zWarning=0)
    AND (za.targetType='SCIENCE' AND za.survey='sdss')
    AND (za.class='GALAXY' AND zi.primtarget>=64)
    AND (po.clean=1 AND po.insideMask=0)
  AND ((po.petroMag_r-po.extinction_r)<=17.8)
  AND za.z <= 0.4
