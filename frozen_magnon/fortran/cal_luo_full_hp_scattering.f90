program cal_luo_full_hp_scattering
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_positive_inf
    !$ use omp_lib
    implicit none

    integer, parameter :: dp = kind(0d0)
    integer, parameter :: nt = 4, nb = 2
    real(dp), parameter :: pi = acos(-1.0d0)
    real(dp), parameter :: kb_mev_k = 8.617333262d-2
    real(dp), parameter :: degeneracy_tolerance_mev = 1.0d-10
    complex(dp), parameter :: ci = (0.0d0,1.0d0)
    ! V contains all Bose contractions; identical beta final states are unordered.
    real(dp), parameter :: final_state_factor(4) = (/0.5d0,0.5d0,1.0d0,1.0d0/)

    integer :: nqx, nqy, nq, iq_start, iq_end
    integer :: iq, ik2, ik3, ik1, id1, id2, it
    integer, allocatable :: qx_i(:), qy_i(:)
    real(dp) :: spin_s, j1, j2, j3, aniso_a, xi_mev, hbar_mev_s
    real(dp) :: temperatures(nt), mesh_norm, prefactor
    real(dp) :: sc_tolerance, sc_mixing
    integer :: sc_max_iterations, sc_nqx, sc_nqy
    logical :: compute_rates
    real(dp), allocatable :: qx(:), qy(:), e_beta_t0(:), e_alpha_t0(:), phase_b_t0(:)
    real(dp), allocatable :: e_beta(:,:), e_alpha(:,:), phase_b(:,:)
    real(dp), allocatable :: g2(:), occ(:,:,:)
    real(dp), allocatable :: rates(:,:,:), full_rates(:,:,:)
    real(dp), allocatable :: magnetization(:), sc_f2(:), sc_residual(:)
    integer, allocatable :: sc_iterations(:)
    complex(dp), allocatable :: sc_f1(:), sc_f3(:)
    complex(dp), allocatable :: g1(:), g3(:), hp_inter(:), hp_phase(:,:)
    complex(dp) :: vertices(4)
    real(dp) :: de(4), delta_l(4), bose(4), n1, n2, n3
    character(len=240) :: rate_file, band_file, diag_file, summary_file, run_class
    character(len=240) :: scattering_definition, spectral_file
    character(len=240) :: dispersion_mode

    call defaults()
    call read_input()
    call build_mesh()
    call build_zero_temperature_bands()
    call build_temperature_bands()
    call prepare_hp_vertices()

    if (compute_rates) then
      if (trim(scattering_definition)=='full_sunset') then
        call calculate_full_sunset()
      else
        allocate(rates(nt,nq,4))
        rates = 0.0d0
        mesh_norm = 1.0d0/real(nq,dp)**2
        prefactor = 2.0d0*pi/hbar_mev_s*mesh_norm

        !$omp parallel do default(shared) schedule(dynamic) &
        !$omp private(ik2,ik3,ik1,id1,id2,it,vertices,de,delta_l,bose,n1,n2,n3)
        do iq=iq_start,iq_end
            do ik2=1,nq
                id1 = q_index(modulo(qx_i(iq)-qx_i(ik2),nqx), &
                              modulo(qy_i(iq)-qy_i(ik2),nqy))
                do ik3=1,nq
                    ik1 = q_index(modulo(qx_i(ik3)+qx_i(iq)-qx_i(ik2),nqx), &
                                  modulo(qy_i(ik3)+qy_i(iq)-qy_i(ik2),nqy))
                    id2 = q_index(modulo(qx_i(ik3)-qx_i(ik2),nqx), &
                                  modulo(qy_i(ik3)-qy_i(ik2),nqy))

                    do it=1,nt
                        call luo_vertices(it,iq,ik1,ik2,ik3,id1,id2,vertices)
                        de(1) = e_beta(it,ik1)+e_beta(it,ik2)-e_beta(it,ik3)-e_beta(it,iq)
                        de(2) = e_beta(it,ik1)+e_beta(it,ik2)-e_beta(it,ik3)-e_alpha(it,iq)
                        de(3) = e_beta(it,ik1)+e_alpha(it,ik2)-e_beta(it,ik3)-e_alpha(it,iq)
                        de(4) = e_alpha(it,ik1)+e_beta(it,ik2)-e_beta(it,ik3)-e_beta(it,iq)
                        delta_l = lorentzian(de,xi_mev)

                        ! Diagonal H4 terms change the real magnon energy but
                        ! cannot broaden a level. Exclude every f=i Fock state.
                        if (same_unordered_pair(ik1,ik2,iq,ik3)) delta_l(1)=0.0d0
                        if (ik1 == ik3 .and. ik2 == iq) delta_l(3)=0.0d0

                        ! At a Dirac degeneracy alpha and beta are an arbitrary
                        ! basis of one subspace. Zero-energy branch conversion
                        ! with an unchanged spectator is not a decay channel.
                        if (abs(e_alpha(it,iq)-e_beta(it,iq)) < degeneracy_tolerance_mev) then
                            if ((ik1 == iq .and. ik2 == ik3) .or. &
                                (ik2 == iq .and. ik1 == ik3)) delta_l(2)=0.0d0
                        endif
                        if ((ik1 == iq .and. ik2 == ik3 .and. &
                             abs(e_alpha(it,iq)-e_beta(it,iq)) < degeneracy_tolerance_mev) .or. &
                            (ik1 == ik3 .and. ik2 == iq .and. &
                             abs(e_alpha(it,ik3)-e_beta(it,ik3)) < degeneracy_tolerance_mev)) then
                            delta_l(4)=0.0d0
                        endif

                    n1 = occ(it,ik1,1); n2 = occ(it,ik2,1); n3 = occ(it,ik3,1)
                    bose(1) = (1.0d0+n1)*(1.0d0+n2)*n3 &
                            - n1*n2*(1.0d0+n3)
                    bose(2) = (1.0d0+n1)*(1.0d0+n2)*n3

                    n1 = occ(it,ik1,1); n2 = occ(it,ik2,2); n3 = occ(it,ik3,1)
                    bose(3) = (1.0d0+n1)*(1.0d0+n2)*n3 &
                            - n1*n2*(1.0d0+n3)

                    n1 = occ(it,ik1,2); n2 = occ(it,ik2,1); n3 = occ(it,ik3,1)
                    bose(4) = (1.0d0+n1)*(1.0d0+n2)*n3

                    rates(it,iq,1) = rates(it,iq,1) &
                            + final_state_factor(1)*abs(vertices(1))**2*bose(1)*delta_l(1)
                    rates(it,iq,2) = rates(it,iq,2) &
                            + final_state_factor(2)*abs(vertices(2))**2*bose(2)*delta_l(2)
                    rates(it,iq,3) = rates(it,iq,3) &
                            + final_state_factor(3)*abs(vertices(3))**2*bose(3)*delta_l(3)
                    rates(it,iq,4) = rates(it,iq,4) &
                            + final_state_factor(4)*abs(vertices(4))**2*bose(4)*delta_l(4)
                    enddo
                enddo
            enddo
            rates(:,iq,:) = prefactor*rates(:,iq,:)
        enddo
        !$omp end parallel do
      endif
    endif

    call write_outputs()

contains

    subroutine defaults()
        nqx = 6; nqy = 6
        iq_start = 1; iq_end = 0
        spin_s = 1.5d0
        j1 = -5.513366996561d0
        j2 = -5.799651014926d0
        j3 = 0.2366628886773d0
        aniso_a = 0.3627d0
        xi_mev = abs(j1)/100.0d0
        hbar_mev_s = 6.582119569d-13
        temperatures = (/6.7d0,20.0d0,30.0d0,50.0d0/)
        dispersion_mode = 'fixed_zero_temperature'
        sc_tolerance = 1.0d-10
        sc_mixing = 0.5d0
        sc_max_iterations = 10000
        sc_nqx = 24; sc_nqy = 24
        compute_rates = .true.
        scattering_definition = 'legacy_four_channel'
        spectral_file = 'full_sunset_spectral.tsv'
        rate_file = 'luo_four_rates.tsv'
        band_file = 'luo_band_t0.tsv'
        diag_file = 'luo_vertex_diagnostic.tsv'
        summary_file = 'luo_scattering_summary.out'
        run_class = 'DIAGNOSTIC_ALIAS_CONTAMINATED'
    end subroutine defaults

    subroutine read_input()
        integer :: unit, io, pos
        character(len=600) :: line
        character(len=240) :: key, value

        open(newunit=unit,file='luo_scatt.in',status='old',action='read')
        do
            read(unit,'(A)',iostat=io) line
            if (io /= 0) exit
            line = adjustl(line)
            if (len_trim(line) == 0 .or. line(1:1) == '#') cycle
            pos = index(line,'=')
            if (pos <= 1) error stop 'bad luo_scatt.in line'
            key = adjustl(line(:pos-1)); value = adjustl(line(pos+1:))
            select case(trim(key))
            case('nqx'); read(value,*) nqx
            case('nqy'); read(value,*) nqy
            case('iq_start'); read(value,*) iq_start
            case('iq_end'); read(value,*) iq_end
            case('spin_S'); read(value,*) spin_s
            case('J1_meV'); read(value,*) j1
            case('J2_meV'); read(value,*) j2
            case('J3_meV'); read(value,*) j3
            case('A_meV'); read(value,*) aniso_a
            case('xi_meV'); read(value,*) xi_mev
            case('hbar_meV_s'); read(value,*) hbar_mev_s
            case('temperatures_K'); read(value,*) temperatures
            case('dispersion_mode'); dispersion_mode = trim(value)
            case('sc_tolerance'); read(value,*) sc_tolerance
            case('sc_mixing'); read(value,*) sc_mixing
            case('sc_max_iterations'); read(value,*) sc_max_iterations
            case('sc_nqx'); read(value,*) sc_nqx
            case('sc_nqy'); read(value,*) sc_nqy
            case('compute_rates'); read(value,*) compute_rates
            case('scattering_definition'); scattering_definition=trim(value)
            case('spectral_file'); spectral_file=trim(value)
            case('rate_file'); rate_file = trim(value)
            case('band_file'); band_file = trim(value)
            case('diag_file'); diag_file = trim(value)
            case('summary_file'); summary_file = trim(value)
            case('run_class'); run_class = trim(value)
            case default; error stop 'unknown luo_scatt.in key'
            end select
        enddo
        close(unit)

        if (nqx <= 1 .or. nqy <= 1) error stop 'nqx/nqy must exceed one'
        if (trim(scattering_definition)/='legacy_four_channel' .and. &
            trim(scattering_definition)/='full_sunset') error stop 'invalid scattering definition'
        if (spin_s <= 0.0d0 .or. aniso_a < 0.0d0) error stop 'invalid spin model'
        if (xi_mev <= 0.0d0 .or. hbar_mev_s <= 0.0d0) error stop 'invalid FGR scale'
        if (any(temperatures <= 0.0d0)) error stop 'temperatures must be positive'
        if (trim(dispersion_mode) /= 'fixed_zero_temperature' .and. &
            trim(dispersion_mode) /= 'self_consistent_mean_field') then
            error stop 'dispersion_mode must be fixed_zero_temperature or self_consistent_mean_field'
        endif
        if (sc_tolerance <= 0.0d0 .or. sc_max_iterations < 1) error stop 'invalid self-consistency controls'
        if (sc_mixing <= 0.0d0 .or. sc_mixing > 1.0d0) error stop 'sc_mixing must be in (0,1]'
        if (sc_nqx <= 1 .or. sc_nqy <= 1) error stop 'sc_nqx/sc_nqy must exceed one'
    end subroutine read_input

    subroutine build_mesh()
        integer :: ix, iy, idx
        nq = nqx*nqy
        if (iq_end == 0) iq_end = nq
        if (iq_start < 1 .or. iq_end > nq .or. iq_start > iq_end) then
            error stop 'invalid iq shard bounds'
        endif
        allocate(qx_i(nq),qy_i(nq),qx(nq),qy(nq))
        idx = 0
        do ix=0,nqx-1
            do iy=0,nqy-1
                idx = idx+1
                qx_i(idx)=ix; qy_i(idx)=iy
                qx(idx)=real(ix,dp)/real(nqx,dp)
                qy(idx)=real(iy,dp)/real(nqy,dp)
            enddo
        enddo
    end subroutine build_mesh

    integer function q_index(ix,iy)
        integer, intent(in) :: ix,iy
        q_index = modulo(ix,nqx)*nqy + modulo(iy,nqy) + 1
    end function q_index

    pure logical function same_unordered_pair(a,b,c,d)
        integer, intent(in) :: a,b,c,d
        same_unordered_pair=(a == c .and. b == d) .or. (a == d .and. b == c)
    end function same_unordered_pair

    subroutine build_zero_temperature_bands()
        integer :: i
        complex(dp) :: b
        real(dp) :: diagonal
        allocate(g1(nq),g3(nq),g2(nq))
        allocate(e_beta_t0(nq),e_alpha_t0(nq),phase_b_t0(nq))
        do i=1,nq
            g1(i) = gamma_inter(qx(i),qy(i),1)
            g3(i) = gamma_inter(qx(i),qy(i),3)
            g2(i) = gamma_second(qx(i),qy(i))
            diagonal = -spin_s*(3.0d0*j1+6.0d0*j2+3.0d0*j3) &
                    + spin_s*6.0d0*j2*g2(i) + (2.0d0*spin_s-1.0d0)*aniso_a
            b = spin_s*(3.0d0*j1*g1(i)+3.0d0*j3*g3(i))
            e_beta_t0(i)=diagonal-abs(b)
            e_alpha_t0(i)=diagonal+abs(b)
            if (abs(b) > 1.0d-14) then
                phase_b_t0(i)=atan2(aimag(b),real(b))
            else
                phase_b_t0(i)=0.0d0
            endif
            if (e_beta_t0(i) <= 0.0d0 .or. e_alpha_t0(i) <= 0.0d0) then
                error stop 'non-positive zero-temperature magnon energy'
            endif
        enddo
    end subroutine build_zero_temperature_bands

    complex(dp) function gamma_inter(x,y,shell)
        real(dp), intent(in) :: x,y
        integer, intent(in) :: shell
        integer :: m, rx(3), ry(3)
        real(dp) :: angle
        if (shell == 1) then
            rx=(/0,-1,0/); ry=(/0,0,1/)
        else
            rx=(/-1,-1,1/); ry=(/-1,1,1/)
        endif
        gamma_inter=(0.0d0,0.0d0)
        do m=1,3
            angle=2.0d0*pi*(x*real(rx(m),dp)+y*real(ry(m),dp))
            gamma_inter=gamma_inter+exp(ci*angle)/3.0d0
        enddo
    end function gamma_inter

    real(dp) function gamma_second(x,y)
        real(dp), intent(in) :: x,y
        gamma_second=(cos(2.0d0*pi*x)+cos(2.0d0*pi*y) &
                +cos(2.0d0*pi*(x+y)))/3.0d0
    end function gamma_second

    subroutine build_temperature_bands()
        integer :: t,i,iteration,ix,iy,idx,sc_nq
        real(dp) :: m,m_new,f2_value,f2_new,residual
        complex(dp) :: f1_value,f1_new,f3_value,f3_new,phase_factor
        real(dp), allocatable :: trial_beta(:),trial_alpha(:),trial_phase(:),sc_g2_values(:),sc_occ(:,:)
        complex(dp), allocatable :: sc_g1_values(:),sc_g3_values(:)

        allocate(e_beta(nt,nq),e_alpha(nt,nq),phase_b(nt,nq))
        allocate(occ(nt,nq,nb))
        allocate(magnetization(nt),sc_f1(nt),sc_f2(nt),sc_f3(nt),sc_iterations(nt),sc_residual(nt))

        if (trim(dispersion_mode) == 'fixed_zero_temperature') then
            do t=1,nt
                e_beta(t,:)=e_beta_t0
                e_alpha(t,:)=e_alpha_t0
                phase_b(t,:)=phase_b_t0
                do i=1,nq
                    occ(t,i,1)=bose_value(e_beta(t,i),temperatures(t))
                    occ(t,i,2)=bose_value(e_alpha(t,i),temperatures(t))
                enddo
                magnetization(t)=spin_s-sum(occ(t,:,:))/(2.0d0*real(nq,dp))
                sc_f1(t)=(0.0d0,0.0d0); sc_f2(t)=0.0d0; sc_f3(t)=(0.0d0,0.0d0)
                sc_iterations(t)=0; sc_residual(t)=0.0d0
            enddo
            return
        endif

        sc_nq=sc_nqx*sc_nqy
        allocate(trial_beta(sc_nq),trial_alpha(sc_nq),trial_phase(sc_nq))
        allocate(sc_g1_values(sc_nq),sc_g3_values(sc_nq),sc_g2_values(sc_nq),sc_occ(sc_nq,nb))
        idx=0
        do ix=0,sc_nqx-1
            do iy=0,sc_nqy-1
                idx=idx+1
                sc_g1_values(idx)=gamma_inter(real(ix,dp)/real(sc_nqx,dp), &
                        real(iy,dp)/real(sc_nqy,dp),1)
                sc_g3_values(idx)=gamma_inter(real(ix,dp)/real(sc_nqx,dp), &
                        real(iy,dp)/real(sc_nqy,dp),3)
                sc_g2_values(idx)=gamma_second(real(ix,dp)/real(sc_nqx,dp), &
                        real(iy,dp)/real(sc_nqy,dp))
            enddo
        enddo
        do t=1,nt
            m=spin_s
            f1_value=(0.0d0,0.0d0); f2_value=0.0d0; f3_value=(0.0d0,0.0d0)
            residual=huge(1.0d0)
            do iteration=1,sc_max_iterations
                call mean_field_bands_on_arrays(sc_nq,sc_g1_values,sc_g2_values,sc_g3_values, &
                        m,f1_value,f2_value,f3_value,trial_beta,trial_alpha,trial_phase)
                do i=1,sc_nq
                    sc_occ(i,1)=bose_value(trial_beta(i),temperatures(t))
                    sc_occ(i,2)=bose_value(trial_alpha(i),temperatures(t))
                enddo
                m_new=spin_s-sum(sc_occ)/(2.0d0*real(sc_nq,dp))
                f1_new=(0.0d0,0.0d0); f2_new=0.0d0; f3_new=(0.0d0,0.0d0)
                do i=1,sc_nq
                    phase_factor=exp(ci*trial_phase(i))
                    f1_new=f1_new+conjg(sc_g1_values(i))*phase_factor*(sc_occ(i,2)-sc_occ(i,1))
                    f3_new=f3_new+conjg(sc_g3_values(i))*phase_factor*(sc_occ(i,2)-sc_occ(i,1))
                    f2_new=f2_new+sc_g2_values(i)*(sc_occ(i,1)+sc_occ(i,2))
                enddo
                f1_new=f1_new/(2.0d0*real(sc_nq,dp))
                f2_new=f2_new/(2.0d0*real(sc_nq,dp))
                f3_new=f3_new/(2.0d0*real(sc_nq,dp))
                residual=max(abs(m_new-m),abs(f1_new-f1_value),abs(f2_new-f2_value),abs(f3_new-f3_value))
                m=(1.0d0-sc_mixing)*m+sc_mixing*m_new
                f1_value=(1.0d0-sc_mixing)*f1_value+sc_mixing*f1_new
                f2_value=(1.0d0-sc_mixing)*f2_value+sc_mixing*f2_new
                f3_value=(1.0d0-sc_mixing)*f3_value+sc_mixing*f3_new
                if (residual < sc_tolerance) exit
            enddo
            if (residual >= sc_tolerance) error stop 'finite-temperature mean-field did not converge'
            call mean_field_bands(m,f1_value,f2_value,f3_value,e_beta(t,:),e_alpha(t,:),phase_b(t,:))
            do i=1,nq
                occ(t,i,1)=bose_value(e_beta(t,i),temperatures(t))
                occ(t,i,2)=bose_value(e_alpha(t,i),temperatures(t))
            enddo
            magnetization(t)=m; sc_f1(t)=f1_value; sc_f2(t)=f2_value; sc_f3(t)=f3_value
            sc_iterations(t)=iteration; sc_residual(t)=residual
        enddo
        deallocate(trial_beta,trial_alpha,trial_phase,sc_g1_values,sc_g2_values,sc_g3_values,sc_occ)
    end subroutine build_temperature_bands

    subroutine mean_field_bands(m,f1_value,f2_value,f3_value,beta_band,alpha_band,band_phase)
        real(dp), intent(in) :: m,f2_value
        complex(dp), intent(in) :: f1_value,f3_value
        real(dp), intent(out) :: beta_band(nq),alpha_band(nq),band_phase(nq)
        call mean_field_bands_on_arrays(nq,g1,g2,g3,m,f1_value,f2_value,f3_value, &
                beta_band,alpha_band,band_phase)
    end subroutine mean_field_bands

    subroutine mean_field_bands_on_arrays(npoints,g1_values,g2_values,g3_values,m,f1_value, &
            f2_value,f3_value,beta_band,alpha_band,band_phase)
        integer, intent(in) :: npoints
        complex(dp), intent(in) :: g1_values(npoints),g3_values(npoints)
        real(dp), intent(in) :: g2_values(npoints),m,f2_value
        complex(dp), intent(in) :: f1_value,f3_value
        real(dp), intent(out) :: beta_band(npoints),alpha_band(npoints),band_phase(npoints)
        integer :: i
        real(dp) :: diagonal
        complex(dp) :: b

        do i=1,npoints
            diagonal=-3.0d0*j1*(m+real(f1_value))-6.0d0*j2*(m+f2_value) &
                    -3.0d0*j3*(m+real(f3_value))+6.0d0*j2*g2_values(i)*(m+f2_value) &
                    -(2.0d0*spin_s+1.0d0-4.0d0*m)*aniso_a
            b=3.0d0*j1*(m+f1_value)*g1_values(i)+3.0d0*j3*(m+f3_value)*g3_values(i)
            beta_band(i)=diagonal-abs(b)
            alpha_band(i)=diagonal+abs(b)
            if (abs(b) > 1.0d-14) then
                band_phase(i)=atan2(aimag(b),real(b))
            else
                band_phase(i)=0.0d0
            endif
            if (beta_band(i) <= 0.0d0 .or. alpha_band(i) <= 0.0d0) then
                error stop 'non-positive self-consistent magnon energy'
            endif
        enddo
    end subroutine mean_field_bands_on_arrays

    elemental real(dp) function bose_value(energy,temp)
        real(dp), intent(in) :: energy,temp
        real(dp) :: x
        x=energy/(kb_mev_k*temp)
        if (x > 700.0d0) then
            bose_value=0.0d0
        else
            bose_value=1.0d0/(exp(x)-1.0d0)
        endif
    end function bose_value

    subroutine prepare_hp_vertices()
        ! Cache the H2 eigenvector phases and the bare complex exchange sum.
        ! Mean-field m+f factors belong to H2; they do not renormalize H4 here.
        if (.not. allocated(hp_phase)) allocate(hp_phase(nt,nq),hp_inter(nq))
        hp_phase=exp(0.5d0*ci*phase_b)
        hp_inter=3.0d0*(j1*g1+j3*g3)
    end subroutine prepare_hp_vertices

    subroutine luo_vertices(t,k,k1,k2,k3,d1,d2,v)
        integer, intent(in) :: t,k,k1,k2,k3,d1,d2
        complex(dp), intent(out) :: v(4)
        complex(dp) :: z(4), x(4), phase_product, transfer1, transfer2
        real(dp) :: same_sublattice

        ! Legs 1,2 are created; 3,4 are annihilated (leg 4 is k).
        ! This is the Fourier sum of the normal-ordered real-space HP H4,
        ! including all four Bose contractions: Vbar=N*V. Keep complex
        ! g1/g3, since their phases differ when both exchange shells exist.
        z=hp_phase(t,(/k1,k2,k3,k/))
        x=z*z*conjg(hp_inter((/k1,k2,k3,k/)))
        x(3:4)=conjg(x(3:4))
        phase_product=conjg(z(1)*z(2))*z(3)*z(4)
        transfer1=conjg(z(1)*z(4))*z(2)*z(3)*hp_inter(d1)
        transfer2=conjg(z(1)*z(3))*z(2)*z(4)*hp_inter(d2)

        same_sublattice=1.5d0*j2*(2.0d0*g2(d1)+2.0d0*g2(d2) &
                -g2(k1)-g2(k2)-g2(k3)-g2(k))-2.0d0*aniso_a

        v(1)=0.5d0*real(transfer1+transfer2,dp) &
                +0.25d0*real(phase_product*(x(1)+x(2)+x(3)+x(4)),dp) &
                +same_sublattice*real(phase_product,dp)
        v(2)=ci*(0.5d0*aimag(-transfer1+transfer2) &
                +0.25d0*aimag(phase_product*(x(1)+x(2)+x(3)-x(4))) &
                +same_sublattice*aimag(phase_product))
        v(3)=0.5d0*real(transfer1-transfer2,dp) &
                +0.25d0*real(phase_product*(x(1)-x(2)+x(3)-x(4)),dp) &
                +same_sublattice*real(phase_product,dp)
        v(4)=ci*(0.5d0*aimag(transfer1+transfer2) &
                +0.25d0*aimag(phase_product*(-x(1)+x(2)+x(3)+x(4))) &
                +same_sublattice*aimag(phase_product))
    end subroutine luo_vertices



    subroutine sunset_vertices(t,k,k1,k2,k3,d1,d2,v)
        integer, intent(in) :: t,k,k1,k2,k3,d1,d2
        complex(dp), intent(out) :: v(16)
        complex(dp) :: z(4),x(4),p,tr1,tr2,total
        real(dp) :: ss,sgn(4)
        integer :: a,b,c,d,m,prod_s
        z=hp_phase(t,(/k1,k2,k3,k/))
        x=z*z*conjg(hp_inter((/k1,k2,k3,k/)))
        x(3:4)=conjg(x(3:4))
        p=conjg(z(1)*z(2))*z(3)*z(4)
        tr1=conjg(z(1)*z(4))*z(2)*z(3)*hp_inter(d1)
        tr2=conjg(z(1)*z(3))*z(2)*z(4)*hp_inter(d2)
        ss=1.5d0*j2*(2*g2(d1)+2*g2(d2)-g2(k1)-g2(k2)-g2(k3)-g2(k))-2*aniso_a
        do a=0,1
          do b=0,1
            do c=0,1
              do d=0,1
                m=1+8*a+4*b+2*c+d
                sgn=real((/2*a-1,2*b-1,2*c-1,2*d-1/),dp)
                prod_s=nint(product(sgn))
                total=sum(sgn*x)
                v(m)=0.25d0*(sgn(2)*sgn(4)*tr1+sgn(2)*sgn(3)*tr2 &
                       +sgn(1)*sgn(4)*conjg(tr2)+sgn(1)*sgn(3)*conjg(tr1)) &
                     -0.125d0*(p*total+prod_s*conjg(p*total)) &
                     +0.5d0*ss*(p+prod_s*conjg(p))
              enddo
            enddo
          enddo
        enddo
    end subroutine sunset_vertices

    subroutine sunset_contribution(t,k,k1,k2,k3,d1,d2,result)
        integer, intent(in) :: t,k,k1,k2,k3,d1,d2
        real(dp), intent(out) :: result(16)
        complex(dp) :: v(16)
        real(dp) :: en(4),oc(3),de,factor
        integer :: legs(4),bs(4),a,b,c,d,m,j,states(4),swap
        logical :: same,bath_degenerate,spectator_only
        legs=(/k1,k2,k3,k/)
        call sunset_vertices(t,k,k1,k2,k3,d1,d2,v)
        bath_degenerate=abs(e_alpha(t,k3)-e_beta(t,k3)) < degeneracy_tolerance_mev
        do a=0,1
          do b=0,1
            do c=0,1
              do d=0,1
                m=1+8*a+4*b+2*c+d
                bs=(/a,b,c,d/)
                states=legs+nq*bs
                if (states(1)>states(2)) then
                  swap=states(1); states(1)=states(2); states(2)=swap
                endif
                if (states(3)>states(4)) then
                  swap=states(3); states(3)=states(4); states(4)=swap
                endif
                same=all(states(1:2)==states(3:4))
                spectator_only=bath_degenerate .and. &
                  ((k1==k3 .and. k2==k .and. b==d) .or. &
                   (k2==k3 .and. k1==k .and. a==d))
                if (same .or. spectator_only) then
                  result(m)=0.0d0
                  cycle
                endif
                do j=1,4
                  if (bs(j)==0) then
                    en(j)=e_beta(t,legs(j))
                  else
                    en(j)=e_alpha(t,legs(j))
                  endif
                  if (j<=3) oc(j)=occ(t,legs(j),bs(j)+1)
                enddo
                de=en(1)+en(2)-en(3)-en(4)
                factor=oc(3)*(1+oc(1)+oc(2))-oc(1)*oc(2)
                result(m)=0.5d0*abs(v(m))**2*factor*lorentzian(de,xi_mev)
              enddo
            enddo
          enddo
        enddo
    end subroutine sunset_contribution

    subroutine calculate_full_sunset()
        real(dp) :: block(nt,16),part(16),scale
        integer :: k2,k3,k1,d1,d2,t
        allocate(full_rates(nt,nq,16))
        full_rates=0.0d0
        scale=2*pi/hbar_mev_s/real(nq,dp)**2
        do iq=iq_start,iq_end
            
            block=0.0d0
            !$omp parallel do default(shared) schedule(static) &
            !$omp private(k3,k1,d1,d2,t,part) reduction(+:block)
            do k2=1,nq
                d1=q_index(qx_i(iq)-qx_i(k2),qy_i(iq)-qy_i(k2))+0
                do k3=1,nq
                    k1=q_index(qx_i(k3)+qx_i(iq)-qx_i(k2),qy_i(k3)+qy_i(iq)-qy_i(k2))+0
                    d2=q_index(qx_i(k3)-qx_i(k2),qy_i(k3)-qy_i(k2))
                    do t=1,nt
                        call sunset_contribution(t,iq,k1,k2,k3,d1,d2,part)
                        block(t,:)=block(t,:)+part
                    enddo
                enddo
            enddo
            !$omp end parallel do
            full_rates(:,iq,:)=scale*block
        enddo
    end subroutine calculate_full_sunset

    subroutine write_spectral_rates()
        integer :: unit,t,i
        real(dp) :: r(2),gamma(2),tau(2)
        open(newunit=unit,file=trim(spectral_file),status='replace',action='write')
        write(unit,'(A)') '# T_K iq qx qy qz E_alpha_meV E_beta_meV R_alpha_s-1 R_beta_s-1 HWHM_alpha_meV HWHM_beta_meV FWHM_alpha_meV FWHM_beta_meV tau_population_alpha_s tau_population_beta_s R_c0000 R_c0001 R_c0010 R_c0011 R_c0100 R_c0101 R_c0110 R_c0111 R_c1000 R_c1001 R_c1010 R_c1011 R_c1100 R_c1101 R_c1110 R_c1111'
        do t=1,nt
          do i=iq_start,iq_end
            
            r=(/sum(full_rates(t,i,2:16:2)),sum(full_rates(t,i,1:16:2))/)
            if (any(r<0) .or. .not.all(ieee_is_finite(r))) error stop 'invalid full sunset branch total'
            gamma=0.5d0*hbar_mev_s*r
            tau=ieee_value(0.0d0,ieee_positive_inf)
            where(r>0) tau=1.0d0/r
            write(unit,'(*(G0,1X))') temperatures(t),i,qx(i),qy(i),0.0d0,e_alpha(t,i),e_beta(t,i), &
                r,gamma,2*gamma,tau,full_rates(t,i,:)
          enddo
        enddo
        close(unit)
    end subroutine write_spectral_rates

    elemental real(dp) function lorentzian(x,width)
        real(dp), intent(in) :: x,width
        lorentzian=width/(pi*(x*x+width*width))
    end function lorentzian

    subroutine write_outputs()
        integer :: unit,t,i,m,k2d,k3d,k1d,d1d,d2d
        complex(dp) :: vd(4)
        real(dp) :: amin,amax

        open(newunit=unit,file=trim(band_file),status='replace',action='write')
        if (trim(dispersion_mode) == 'fixed_zero_temperature') then
            write(unit,'(A)') '# iq qx qy qz E_alpha_meV E_beta_meV phase_B_rad'
            do i=1,nq
                write(unit,'(I0,1X,6(ES24.15E3,1X))') i,qx(i),qy(i),0.0d0, &
                        e_alpha_t0(i),e_beta_t0(i),phase_b_t0(i)
            enddo
        else
            write(unit,'(A)') '# T_K iq qx qy qz E_alpha_T_meV E_beta_T_meV phase_B_T_rad E_alpha_t0_meV E_beta_t0_meV phase_B_t0_rad M f1_re f1_im f2 f3_re f3_im iterations residual'
            do t=1,nt
                do i=1,nq
                    write(unit,'(ES16.8E3,1X,I0,1X,15(ES24.15E3,1X),I0,1X,ES24.15E3)') &
                            temperatures(t),i,qx(i),qy(i),0.0d0,e_alpha(t,i),e_beta(t,i),phase_b(t,i), &
                            e_alpha_t0(i),e_beta_t0(i),phase_b_t0(i),magnetization(t),real(sc_f1(t)), &
                            aimag(sc_f1(t)),sc_f2(t),real(sc_f3(t)),aimag(sc_f3(t)),sc_iterations(t),sc_residual(t)
                enddo
            enddo
        endif
        close(unit)

        if (compute_rates .and. trim(scattering_definition)=='full_sunset') call write_spectral_rates()
        if (compute_rates .and. trim(scattering_definition)=='legacy_four_channel') then
            open(newunit=unit,file=trim(rate_file),status='replace',action='write')
            write(unit,'(A)') '# T_K iq qx qy qz E_alpha_meV E_beta_meV eta1_s-1 eta2_s-1 eta3_s-1 eta4_s-1 eta_alpha_s-1 eta_beta_s-1'
            do t=1,nt
                do i=iq_start,iq_end
                    if (.not. all(ieee_is_finite(rates(t,i,:)))) error stop 'non-finite rate'
                    write(unit,'(ES16.8E3,1X,I0,1X,11(ES24.15E3,1X))') &
                            temperatures(t),i,qx(i),qy(i),0.0d0,e_alpha(t,i),e_beta(t,i), &
                            rates(t,i,1),rates(t,i,2),rates(t,i,3),rates(t,i,4), &
                            rates(t,i,2)+rates(t,i,3),rates(t,i,1)+rates(t,i,4)
                enddo
            enddo
            close(unit)
        endif

        k2d=q_index(1,0); k3d=q_index(0,1)
        k1d=q_index(modulo(qx_i(k3d)-qx_i(k2d),nqx), &
                    modulo(qy_i(k3d)-qy_i(k2d),nqy))
        d1d=q_index(modulo(-qx_i(k2d),nqx),modulo(-qy_i(k2d),nqy))
        d2d=q_index(modulo(qx_i(k3d)-qx_i(k2d),nqx), &
                    modulo(qy_i(k3d)-qy_i(k2d),nqy))
        open(newunit=unit,file=trim(diag_file),status='replace',action='write')
        write(unit,'(A)') '# T_K m Re_Vbar_meV Im_Vbar_meV abs2_Vbar_meV2 Re_V_meV Im_V_meV'
        do t=1,nt
            call luo_vertices(t,1,k1d,k2d,k3d,d1d,d2d,vd)
            do m=1,4
                write(unit,'(ES16.8E3,1X,I0,1X,5(ES24.15E3,1X))') temperatures(t),m, &
                        real(vd(m)),aimag(vd(m)),abs(vd(m))**2, &
                        real(vd(m))/real(nq,dp),aimag(vd(m))/real(nq,dp)
            enddo
        enddo
        close(unit)

        amin=minval(e_beta); amax=maxval(e_alpha)
        open(newunit=unit,file=trim(summary_file),status='replace',action='write')
        write(unit,'(A)') 'scattering_definition='//trim(scattering_definition)
        if (trim(scattering_definition)=='full_sunset') then
            write(unit,'(A)') 'spectral_definition=ON_SHELL_HWHM_HBAR_NET_RATE_OVER_2'
            write(unit,'(A)') 'branch_combinations=16_ordered_final_half_all'
            write(unit,'(A)') 'bose_factor=n_bath*(1+n_final1+n_final2)-n_final1*n_final2'
            write(unit,'(A)') 'quartic_order=existing_leading_HP_normal_ordered_J123_A'
            write(unit,'(A)') 'reference=Eto_PRX16_021053_2026_Eqs12_15_29'
        endif
        write(unit,'(A)') 'mode=luo_full_hp'
        write(unit,'(A)') 'run_class='//trim(run_class)
        write(unit,'(A)') 'hamiltonian=H_plus_sum_J_SdotS_minus_A_Sz2'
        write(unit,'(A,ES24.15E3)') 'J1_meV=',j1
        write(unit,'(A,ES24.15E3)') 'J2_meV=',j2
        write(unit,'(A,ES24.15E3)') 'J3_meV=',j3
        write(unit,'(A,ES24.15E3)') 'A_meV=',aniso_a
        write(unit,'(A,ES24.15E3)') 'gap_meV=',(2.0d0*spin_s-1.0d0)*aniso_a
        write(unit,'(A,ES24.15E3)') 'xi_meV=',xi_mev
        write(unit,'(A,2(I0,1X))') 'mesh=',nqx,nqy
        write(unit,'(A,2(I0,1X))') 'iq_range=',iq_start,iq_end
        write(unit,'(A,4(ES16.8E3,1X))') 'temperatures_K=',temperatures
        write(unit,'(A,ES24.15E3)') 'omega_min_meV=',amin
        write(unit,'(A,ES24.15E3)') 'omega_max_meV=',amax
        write(unit,'(A)') 'delta=lorentzian_xi_over_pi_x2_plus_xi2'
        write(unit,'(A)') 'fgr_transition_filter=exclude_f_equals_i_and_degenerate_branch_conversion'
        write(unit,'(A,ES24.15E3)') 'degeneracy_tolerance_meV=',degeneracy_tolerance_mev
        write(unit,'(A)') 'dispersion='//trim(dispersion_mode)
        write(unit,'(A,L1)') 'compute_rates=',compute_rates
        write(unit,'(A,ES24.15E3)') 'self_consistency_tolerance=',sc_tolerance
        write(unit,'(A,ES24.15E3)') 'self_consistency_mixing=',sc_mixing
        write(unit,'(A,I0)') 'self_consistency_max_iterations=',sc_max_iterations
        write(unit,'(A,2(I0,1X))') 'self_consistency_mesh=',sc_nqx,sc_nqy
        do t=1,nt
            write(unit,'(A,ES12.4E3,A,ES24.15E3,A,I0,A,ES24.15E3)') 'T=',temperatures(t), &
                    ' M=',magnetization(t),' iterations=',sc_iterations(t),' residual=',sc_residual(t)
            write(unit,'(A,6(ES24.15E3,1X))') 'correlations_f1_re_im_f2_f3_re_im=', &
                    real(sc_f1(t)),aimag(sc_f1(t)),sc_f2(t),real(sc_f3(t)),aimag(sc_f3(t))
        enddo
        write(unit,'(A)') 'vertices=HP_COMPLEX_BONDS_V1_Vbar_equals_N_times_V'
        if (trim(scattering_definition)=='legacy_four_channel') then
        write(unit,'(A)') 'fgr_counting=ORDERED_FINAL_HALF_IDENTICAL_V1'
        write(unit,'(A,4(F4.1,1X))') 'fgr_final_state_factors=',final_state_factor
        else
        write(unit,'(A)') 'fgr_counting=ORDERED_FINAL_HALF_ALL_BRANCHES_V1'
        write(unit,'(A)') 'fgr_final_state_factors=0.5_for_all_16'
        endif
        close(unit)
    end subroutine write_outputs

end program cal_luo_full_hp_scattering
