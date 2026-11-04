program extract
  use parameter_mod
        !! This code convert the V_vrs from QE to 
        !! our format in the real-space index of the subspace
        !! chosen by R_sym.f90, with symmetry operation.

        !! Note that the details need to be modify for other symmetry operation
        !! Note that some variables may looks redundant.
        !! They are kept for future extension.

        implicit none
        integer :: nx,ny,nz,n_file,i_file,i_iter,n_iter,i,j
        integer :: ix,iy,iz,jx,jy,jz,n_sym
        integer :: i_sym,ni,iz_site,iy_site,i_site,ix_site
        integer :: super_x,super_y,super_z
        integer :: i_atom, n_atom
        real(kind(0d0)),allocatable :: read_V(:,:)
        real(kind(0d0)),allocatable :: V_write(:,:)
        real(kind(0d0)),allocatable :: V1(:,:)
        character(len=20) :: file_char,i_sym_char,nspinor_char
        real(kind(0d0))    :: read4(4),pi
        real(kind(0d0)),allocatable :: spin(:),spin_all(:,:)
        integer ,allocatable    :: R_vec(:,:)
        character(len=20) :: formatname
        integer :: z_box,io,nspinor,n_dep
        real:: r
        real,allocatable    :: angle_file(:,:),to_write(:,:)
        real,allocatable    :: amp(:),theta(:),phi(:)

        integer,allocatable :: sym_BZ(:,:)

        pi=3.1415926d0
        n_dep=p_split_nspinor
        i_atom=0
        open(unit=101,file='super_atom',action='read'&
                                &,status='old',access='sequential')
        do
                read(101,'(A14,3F20.10)',iostat=io)
                if(io/=0) exit
                i_atom=i_atom+1
        enddo
        close(101)
        n_atom=i_atom

        allocate(spin_all(n_atom,3))
        allocate(amp(n_atom),theta(n_atom),phi(n_atom))
        open(unit=101,file='./polar2',action='read',status='old',access='sequential')
        do i_atom=1,n_atom
                read(101,*) amp(i_atom),theta(i_atom),phi(i_atom)
                spin_all(i_atom,1)=amp(i_atom)*sin(pi/180*theta(i_atom))*cos(pi/180*phi(i_atom))
                spin_all(i_atom,2)=amp(i_atom)*sin(pi/180*theta(i_atom))*sin(pi/180*phi(i_atom))
                spin_all(i_atom,3)=amp(i_atom)*cos(pi/180*theta(i_atom))
        enddo
        close(101)

        

        super_x=p_split_super_x
        super_y=p_split_super_y
        super_z=p_split_super_z
        allocate(spin(n_dep*3))
        allocate(to_write(n_dep,3))


        open(unit=101,file='./V_HYC_dp',access='sequential',action='read'&
                &,form='unformatted')
                read(101) nx,ny,nz
                print*,'nx ny nz=',nx,ny,nz
                allocate(read_V(nx*ny*nz,4))
                allocate(V1((nx/super_x+1)*(ny/super_y+1)*(nz/super_z+1),4))
                read(101) read_V
                close(101)


        i_file=1

        open(unit=203,file='./spin_info',action='write',access='append',status='replace')
        do ix_site=0,super_x-1
        do iy_site=0,super_y-1
        do iz_site=0,super_z-1
                V1(:,:)=0.0d0
                do ix=0,nx/super_x
                do iy=0,ny/super_y
                do iz=0,nz/super_z
                                i=1+ix+(nx/super_x+1)*iy+(nx/super_x+1)*(ny/super_y+1)*iz
                                jx=mod(nx/super_x*ix_site+ix,nx)
                                jy=mod(ny/super_y*iy_site+iy,ny)
                                jz=mod(nz/super_z*iz_site+iz,nz)
                                j=1+jx+nx*jy+nx*ny*jz
                                V1(i,:)=read_V(j,:)
                enddo
                enddo
                enddo

                call int2char(i_file,file_char)

                open(unit=103,file='./V'//trim(file_char),action='write'&
                        &, access='append',status='replace',form='unformatted')
                        write(103) (nx+1)*(ny+1)*(nz+1)/(super_x*super_y*super_z)
                        write(103) V1
                close(103)


                if(iy_site .eq. 0) then
                        to_write(1,:)= spin_all(1,:)   ! 
                        to_write(2,:)= spin_all(2,:)   ! 
                endif



                call int2char(n_dep*3,nspinor_char)
                formatname='('//trim(nspinor_char)//'F12.6)'


                !write(202,formatname) (to_write(i,:),i=1,n_dep)

                do i=1,n_dep

                        spin(1+(i-1)*3)=to_write(i,1)!to_write(i,1)*sin(2*pi/360*to_write(i,2))*cos(2*pi/360*to_write(i,3))
                        spin(2+(i-1)*3)=to_write(i,2)!*sin(2*pi/360*to_write(i,2))*sin(2*pi/360*to_write(i,3))
                        spin(3+(i-1)*3)=to_write(i,3)!*cos(2*pi/360*to_write(i,2))
                enddo
                write(203,formatname) spin(:)

                i_file=i_file+1


        enddo
        enddo
        enddo
        close(203)




contains

subroutine int2char(in_int,out_char)
        integer,intent(in)              :: in_int
        character(len=20),intent(out)    :: out_char

IF (in_int .LT. 10 )            WRITE(out_char,'(I1)')in_int
IF ((in_int .LT. 100)  .and. (in_int .GE. 10)   )WRITE(out_char,'(I2)')in_int
IF ((in_int .LT. 1000) .and. (in_int .GE. 100)  )WRITE(out_char,'(I3)')in_int
IF ((in_int .LT. 10000) .and. (in_int .GE. 1000) )WRITE(out_char,'(I4)')in_int
IF ((in_int .LT. 100000) .and. (in_int .GE. 10000))WRITE(out_char,'(I5)')in_int

end subroutine int2char

end program
