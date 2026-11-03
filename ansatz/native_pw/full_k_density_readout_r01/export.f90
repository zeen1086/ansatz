PROGRAM frozen_spin_coordinates
  USE kinds, ONLY: DP
  USE io_global, ONLY: ionode, ionode_id, stdout
  USE io_files, ONLY: tmp_dir, prefix, restart_dir
  USE mp_global, ONLY: mp_startup
  USE mp_images, ONLY: intra_image_comm
  USE mp, ONLY: mp_bcast, mp_sum
  USE environment, ONLY: environment_start, environment_end
  USE fft_base, ONLY: dfftp
  USE lsda_mod, ONLY: nspin
  USE scf, ONLY: rho
  USE klist, ONLY: nks, nkstot, wk
  USE wvfct, ONLY: wg, et
  USE wavefunctions, ONLY: evc
  USE pw_restart_new, ONLY: read_collected_wfc
  USE gvecs, ONLY: doublegrid
  USE ions_base, ONLY: nat
  USE cell_base, ONLY: omega
  USE mp_bands, ONLY: intra_bgrp_comm
  USE noncollin_module, ONLY: pointlist, factlist
  USE scatter_mod, ONLY: gather_grid
  IMPLICIT NONE
  LOGICAL :: needwf
  INTEGER :: ios, npoint, u, j, ik
  REAL(DP), ALLOCATABLE :: full_rho(:,:)

  CHARACTER(LEN=256) :: outdir
  CHARACTER(LEN=64) :: grid_mode
  CHARACTER(LEN=256), EXTERNAL :: trimcheck
  REAL(DP), ALLOCATABLE :: local_charge(:), moments(:,:), seed_charge(:), seed_moments(:,:)
  REAL(DP) :: total_charge
  NAMELIST /inputpp/ outdir, prefix
  CALL mp_startup()
  CALL environment_start('CRI3_VARIATIONAL_FIELD_EXPORT')
  IF (ionode) CALL input_from_file()
  prefix = 'pwscf'
  CALL get_environment_variable('ESPRESSO_TMPDIR', outdir)
  IF (TRIM(outdir) == '') outdir = './'
  ios = 0
  IF (ionode) THEN
     READ(5, inputpp, iostat=ios)
     tmp_dir = trimcheck(outdir)
  END IF
  CALL mp_bcast(ios, ionode_id, intra_image_comm)
  IF (ios /= 0) CALL errore('cri3_unprojected_gradrot', 'reading inputpp', ABS(ios))
  CALL mp_bcast(tmp_dir, ionode_id, intra_image_comm)
  CALL mp_bcast(prefix, ionode_id, intra_image_comm)

  needwf = .TRUE.
  CALL read_file_new(needwf)
  IF (nspin /= 4) CALL errore('cri3_unprojected_gradrot', 'requires nspin=4', 1)
  npoint = dfftp%nr1x*dfftp%nr2x*dfftp%nr3x
  IF (doublegrid) CALL errore('hf_torque','requires matching smooth and dense grids',1)
  grid_mode = 'unprojected_rebuilt'

  IF(nks/=nkstot .OR. ABS(SUM(wk)-1.0_DP)>1.e-10_DP) &
    CALL errore('spin_coordinates','requires one pool normalized weights',1)
  IF(MINVAL(et(211,:))-MAXVAL(et(210,:))<=0.0_DP) &
    CALL errore('spin_coordinates','occupied gap missing',1)
  IF (.NOT.ALLOCATED(pointlist)) ALLOCATE(pointlist(dfftp%nnr))
  IF (.NOT.ALLOCATED(factlist)) ALLOCATE(factlist(dfftp%nnr))
  CALL make_pointlists()
  ALLOCATE(local_charge(nat),moments(3,nat),seed_charge(nat),seed_moments(3,nat))
  CALL get_locals(seed_charge,seed_moments,rho%of_r)
  CALL read_collected_wfc(restart_dir(),1,evc)
  CALL sum_band()
  DO ik=1,nks
    IF (MAXVAL(ABS(wg(1:210,ik)-wk(ik)))>1.e-10_DP .OR. &
      MAXVAL(ABS(wg(211:,ik)))>1.e-10_DP) CALL errore('spin_coordinates','noninteger occupancy',1)
  END DO
  total_charge=SUM(rho%of_r(:,1))*omega/(dfftp%nr1*dfftp%nr2*dfftp%nr3)
  CALL mp_sum(total_charge,intra_bgrp_comm)
  IF(ABS(total_charge-210.0_DP)>1.e-5_DP) CALL errore('spin_coordinates','charge normalization',1)
  CALL get_locals(local_charge,moments,rho%of_r)
  ALLOCATE(full_rho(npoint,4))
  DO j=1,4
    CALL gather_grid(dfftp,rho%of_r(:,j),full_rho(:,j))
  END DO
  IF(ionode) THEN
    OPEN(NEWUNIT=u,FILE='RHO_OUT.bin',STATUS='new',FORM='unformatted')
    WRITE(u) dfftp%nr1x,dfftp%nr2x,dfftp%nr3x,4
    WRITE(u) full_rho
    CLOSE(u)
    OPEN(NEWUNIT=u,FILE='LOCAL_MOMENTS.dat',STATUS='new')
    DO j=1,nat
      WRITE(u,'(I5,8ES24.15)') j,local_charge(j),moments(:,j),seed_charge(j),seed_moments(:,j)
    END DO
    CLOSE(u)
    WRITE(stdout,'(A,ES24.15)') 'REBUILT_CHARGE ',total_charge
    WRITE(stdout,'(A)') 'FROZEN_SPIN_COORDINATES_COMPLETE'
  END IF
  CALL environment_end('CRI3_VARIATIONAL_FIELD_EXPORT')
  CALL stop_pp()
END PROGRAM frozen_spin_coordinates
