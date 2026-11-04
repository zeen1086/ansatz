PROGRAM cri3_unprojected_gradrot
  USE kinds,             ONLY : DP
  USE io_global,         ONLY : ionode, ionode_id, stdout
  USE io_files,          ONLY : tmp_dir, prefix
  USE mp_global,         ONLY : mp_startup
  USE mp_images,         ONLY : intra_image_comm
  USE mp,                ONLY : mp_bcast
  USE environment,       ONLY : environment_start, environment_end
  USE fft_base,          ONLY : dfftp
  USE fft_rho,           ONLY : rho_r2g
  USE lsda_mod,          ONLY : nspin
  USE scf,               ONLY : rho, rho_core, rhog_core, v
  USE noncollin_module,  ONLY : i_cons
  USE cell_base, ONLY: omega
  USE ener, ONLY: etxcc
  USE ldaU,              ONLY : eth
  IMPLICIT NONE

  LOGICAL :: exists, needwf
  INTEGER :: ios, npoint, output_unit, density_shape(4)
  CHARACTER(LEN=256) :: outdir
  CHARACTER(LEN=64) :: case_name, grid_mode
  CHARACTER(LEN=256), EXTERNAL :: trimcheck
  REAL(DP) :: ehart, etxc, vtxc, etotefield, charge
  REAL(DP), ALLOCATABLE :: input_rho(:,:), core_r(:,:)
  COMPLEX(DP), ALLOCATABLE :: core_g(:,:)
  NAMELIST / inputpp / outdir, prefix

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

  needwf = .FALSE.
  CALL read_file_new(needwf)
  IF (nspin /= 4) CALL errore('cri3_unprojected_gradrot', 'requires nspin=4', 1)
  npoint = dfftp%nr1x*dfftp%nr2x*dfftp%nr3x
  IF (dfftp%nnr /= npoint) CALL errore('cri3_unprojected_gradrot', 'requires one rank', 1)
  grid_mode = 'unprojected_rebuilt'

  OPEN(NEWUNIT=output_unit,FILE='DENSITY_ANSATZ.bin',STATUS='old',FORM='unformatted')
  READ(output_unit) density_shape
  IF (ANY(density_shape/=[dfftp%nr1x,dfftp%nr2x,dfftp%nr3x,nspin])) &
    CALL errore('density_ansatz','grid mismatch',1)
  READ(output_unit) rho%of_r
  CLOSE(output_unit)
  CALL rho_r2g(dfftp, rho%of_r, rho%of_g)
  ALLOCATE(core_r(dfftp%nnr,1), core_g(SIZE(rhog_core),1))
  core_r(:,1) = rho_core
  CALL rho_r2g(dfftp, core_r, core_g)
  rhog_core(:) = core_g(:,1)
  DEALLOCATE(core_r, core_g)

  i_cons = 0
  etotefield = 0.0_DP
  ALLOCATE(input_rho(dfftp%nnr,nspin))
  input_rho=rho%of_r
  CALL v_of_rho(rho, rho_core, rhog_core, ehart, etxc, vtxc, eth, &
                etotefield, charge, v)
  OPEN(NEWUNIT=output_unit, FILE='FROZEN_HXC.bin', STATUS='new', FORM='unformatted')
  WRITE(output_unit) dfftp%nr1x, dfftp%nr2x, dfftp%nr3x, nspin
  WRITE(output_unit) v%of_r
  CLOSE(output_unit)
  OPEN(NEWUNIT=output_unit, FILE='ENERGY_COMPONENTS.dat', STATUS='new')
  WRITE(output_unit,'(A)') '# ehart etxc etxcc U_HXC charge vtxc; energy Ry'
  WRITE(output_unit,'(6ES26.17)') ehart,etxc,etxcc, &
       SUM(input_rho*v%of_r)*omega/(dfftp%nr1*dfftp%nr2*dfftp%nr3),charge,vtxc
  CLOSE(output_unit)
  WRITE(stdout,'(5X,A)') 'CRI3_VARIATIONAL_FIELD_EXPORT_COMPLETE'
  CALL environment_end('CRI3_VARIATIONAL_FIELD_EXPORT')
  CALL stop_pp()
END PROGRAM cri3_unprojected_gradrot
