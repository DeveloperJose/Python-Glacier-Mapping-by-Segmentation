
;---------------------------------------------------------------------------
;                       FILL THE SLC-OFF GAP OF ETM+
;                    Using multiple SLC-off ETM+ images
;     VERSION 2: can be used for whole ETM scene,modify the bug in version 1
;            Developed by Zhu Xiaolin,email: zhuxiaolin.rs@gmail.com
;             Beijing Normal University,Ohio State University
;                    Dr.Jin Chen, email:chenjin@ires.cn
;                       Beijing Normal University
;                                  2010-8-1
;                      Copyright belong to Chen's Lab
;---------------------------------------------------------------------------


;function for open the file

Pro GetData,ImgData = ImgData,ns = ns,nl = nl,nb = nb,Data_Type = Data_Type,$
    FileName = FileName,Map_info = map_Info, Fid = Fid
    Filter = ['all file;*.*']
    Envi_Open_File,FileName,R_Fid = Fid
    Envi_File_Query,Fid,ns = ns,nl = nl,nb = nb,Data_Type = Data_Type
    map_info = envi_get_map_info(fid=Fid)
    dims = [-1,0,ns - 1 ,0,nl - 1]
    case Data_Type Of
        1:ImgData = BytArr(ns,nl,nb)    ;  BYTE  Byte
        2:ImgData = IntArr(ns,nl,nb)    ;  INT  Integer
        3:ImgData = LonArr(ns,nl,nb)    ;  LONG  Longword integer
        4:ImgData = FltArr(ns,nl,nb)    ;  FLOAT  Floating point
        5:ImgData = DblArr(ns,nl,nb)    ;  DOUBLE  Double-precision floating
        6:ImgData = COMPLEXARR(ns,nl,nb); complex, single-precision, floating-point
        9:ImgData = DCOMPLEXARR(ns,nl,nb);complex, double-precision, floating-point
        12:ImgData = UINTARR(ns,nl,nb)   ; unsigned integer vector or array
        13:ImgData = ULONARR(ns,nl,nb)   ;  unsigned longword integer vector or array
        14:ImgData = LON64ARR(ns,nl,nb)   ;a 64-bit integer vector or array
        15:ImgData = ULON64ARR(ns,nl,nb)   ;an unsigned 64-bit integer vector or array
    EndCase
    For i = 0,nb-1 Do Begin
       Dt = Envi_Get_Data(Fid = Fid,dims = dims,pos=i)
       ImgData[*,*,i] = Dt[*,*]
    EndFor
End

;-------------------------------------------------------------------
;                  main body of the program
;-------------------------------------------------------------------

pro  FILLGAP_MULTIPLE_V2

 t0=systime(1)                  ;the initial time of program running

 ;please set the following parameters
;----------------------------------------------------------------------
 min_similar=20                 ;set the minimum sample size of similar pixels
 max_window=15                  ;set the maximum window size
 num_class=4                    ;set the estimated number of classes
 num_input=2                    ;set the number of input images
 DN_min=0                       ;set the range of DN value of the image,If byte, 0 and 255
 DN_max=1
 patch_long=500                ;set the size of block,if process whole ETM scene, set 1000
 temp_file='G:\temp'            ;set the temporary file location
;------------------------------------------------------------------------


 ;open the SLC-off ETM+ image
  FileName1 = Dialog_PickFile(title = 'Open the SLC-off ETM+ image:')
  envi_open_file,FileName1,r_fid=fid
  envi_file_query,fid,ns=ns,nl=nl,nb=nb,dims=dims
  map_info = envi_get_map_info(fid=fid)
  orig_ns=ns
  orig_nl=nl
  n_ns=ceil(float(ns)/patch_long)
  n_nl=ceil(float(nl)/patch_long)

  ind_patch1=intarr(4,n_ns*n_nl)           ;divide the whole scene into 1000*1000 block
  ind_patch=intarr(4,n_ns*n_nl)
  location=intarr(4,n_ns*n_nl)

  for i_ns=0,n_ns-1,1 do begin
    for i_nl=0,n_nl-1,1 do begin
        ind_patch1[0,n_ns*i_nl+i_ns]=i_ns*patch_long
        ind_patch[0,n_ns*i_nl+i_ns]=max([0,ind_patch1[0,n_ns*i_nl+i_ns]-max_window])
        location[0,n_ns*i_nl+i_ns]=ind_patch1[0,n_ns*i_nl+i_ns]-ind_patch[0,n_ns*i_nl+i_ns]

        ind_patch1[1,n_ns*i_nl+i_ns]=min([ns-1,(i_ns+1)*patch_long-1])
        ind_patch[1,n_ns*i_nl+i_ns]=min([ns-1,ind_patch1[1,n_ns*i_nl+i_ns]+max_window])
        location[1,n_ns*i_nl+i_ns]=ind_patch1[1,n_ns*i_nl+i_ns]-ind_patch1[0,n_ns*i_nl+i_ns]+location[0,n_ns*i_nl+i_ns]

        ind_patch1[2,n_ns*i_nl+i_ns]=i_nl*patch_long
        ind_patch[2,n_ns*i_nl+i_ns]=max([0,ind_patch1[2,n_ns*i_nl+i_ns]-max_window])
        location[2,n_ns*i_nl+i_ns]=ind_patch1[2,n_ns*i_nl+i_ns]-ind_patch[2,n_ns*i_nl+i_ns]

        ind_patch1[3,n_ns*i_nl+i_ns]=min([nl-1,(i_nl+1)*patch_long-1])
        ind_patch[3,n_ns*i_nl+i_ns]=min([nl-1,ind_patch1[3,n_ns*i_nl+i_ns]+max_window])
        location[3,n_ns*i_nl+i_ns]=ind_patch1[3,n_ns*i_nl+i_ns]-ind_patch1[2,n_ns*i_nl+i_ns]+location[2,n_ns*i_nl+i_ns]
    endfor
  endfor

  tempoutname=temp_file+'\temp_target'

  pos=indgen(nb)
  for isub=0,n_ns*n_nl-1,1 do begin
      dims=[-1,ind_patch[0,isub],ind_patch[1,isub],ind_patch[2,isub],ind_patch[3,isub]]
      envi_doit, 'resize_doit', fid=fid, pos=pos, dims=dims, interp=0, rfact=[1,1], $
      out_name=tempoutname+strtrim(isub+1,1), r_fid=r_fid1
      envi_file_mng, id=r_fid1, /remove
  endfor

  envi_file_mng, id=fid, /remove

  ;-----------------------------------------------------------     ;open the multiple input SLC-off ETM+ image
 for i1=1,num_input,1 do begin
   FileName2 = Dialog_PickFile(title = 'Open the input image'+strtrim(i1,1)+':')
   envi_open_file,FileName2,r_fid=fid
   tempoutname=temp_file+'\temp_input'+strtrim(i1,1)
   pos=indgen(nb)
   for isub=0,n_ns*n_nl-1,1 do begin
      dims=[-1,ind_patch[0,isub],ind_patch[1,isub],ind_patch[2,isub],ind_patch[3,isub]]
      envi_doit, 'resize_doit', fid=fid, pos=pos, dims=dims, interp=0, rfact=[1,1], $
      out_name=tempoutname+strtrim(isub+1,1), r_fid=r_fid1
      envi_file_mng, id=r_fid1, /remove
  endfor
   envi_file_mng, id=fid, /remove
 endfor

;------------------------------------------------------------------
        ; begin process the gap for each block
;-------------------------------------------------------------------
tempoutname1=temp_file+'\temp_filled'
tempoutname2=temp_file+'\temp_mask'

print,'there are total',n_ns*n_nl,' blocks'

for isub=0,n_ns*n_nl-1,1 do begin

;open each block image

    FileName = temp_file+'\temp_target'
    GetData,ImgData=ImgData,ns = ns,nl = nl,nb = nb,Data_Type = Data_Type,FileName = FileName+strtrim(isub+1,1),Fid = Fid1
    fine1=float(ImgData)
    fine0=fine1[location[0,isub]:location[1,isub],location[2,isub]:location[3,isub],*]    ;place the new image value

    gap=intarr(ns,nl)
    for i=0,ns-1,1 do begin
       for j=0,nl-1,1 do begin
          ind_gap=where(fine1[i,j,*] eq 0, c_gap_band)
          if (c_gap_band gt 0) then begin
             up=max([0,j-15])
             dowm=min([nl-1,j+15])
             ind_up=where(fine1[i,up,*] ne 0, c_gap_up)
             ind_dowm=where(fine1[i,dowm,*] ne 0, c_gap_down)
             if (c_gap_up gt 0 or c_gap_down gt 0) then begin
                gap[i,j]=1
             endif
          endif
       endfor
    endfor
   mark=BYTARR(ns,nl)     ;place the mark value
   note_finish=0
   i_input=1

 while (note_finish ne 1 and i_input le num_input) do begin

       FileName = temp_file+'\temp_input'+strtrim(i_input,1)
       GetData,ImgData=ImgData,FileName = FileName+strtrim(isub+1,1),Fid = Fid2
       fine2=float(ImgData)
       ImgData=0 ;clear this variable

       similar_th_band=fltarr(nb)
        for iband=0,nb-1,1 do begin
          ind_nogrand=where(fine2[*,*,iband] ne 0,c_nogrand)
          if (c_nogrand gt 0) then begin
            similar_th_band[iband]=stddev((fine2[*,*,iband])[ind_nogrand])*2.0/float(num_class)    ;compute the threshold of similar pixel
          endif
        endfor

       similar_th=mean(similar_th_band)


for i=location[0,isub],location[1,isub],1 do begin
  for j=location[2,isub],location[3,isub],1 do begin

    if (gap[i,j] eq 1) then begin

       ind_slc_on=where (fine2[i,j,*] eq 0,c_slc_on)

       if (c_slc_on eq 0) then begin               ;if the corresponding pixel avalible,use this input image

        gap[i,j]=-1       ;denote the target pixel

        extent=ceil(0.5*(min_similar^0.5-1)) ;compute the minimum window size

        note=0                                  ;use adaptive window seek similar valid pixel around the gap
        while (note ne 1 and extent le max_window) do begin
           a1=max([0,i-extent])
           a2=min([ns-1,i+extent])
           b1=max([0,j-extent])
           b2=min([nl-1,j+extent])
           sub_gap=gap[a1:a2,b1:b2]
           sub_off=fine1[a1:a2,b1:b2,*]
           sub_on=fine2[a1:a2,b1:b2,*]
           validate_fine2=intarr(a2-a1+1,b2-b1+1)
           for ib=0,nb-1,1 do begin
             ind_validata=where(sub_on[*,*,ib] ne 0, c_validate)
             if (c_validate gt 0) then begin
                validate_fine2[ind_validata]=validate_fine2[ind_validata]+1
             endif
           endfor
           ind_common=where(sub_gap eq 0 and validate_fine2 eq nb, c_common)
           ind_target=where(sub_gap eq -1)
           it=ind_target mod (a2-a1+1)
           jt=ind_target/(a2-a1+1)

           if (c_common gt min_similar) then begin

                 rmsei=fltarr(c_common)
                 rmse12=fltarr(c_common)
                 disi=fltarr(c_common)
              for icommon=0l,long(c_common)-1l,1l do begin
                  diff_sq=0.0
                  diff_sq2=0.0
                  for iband=0,nb-1,1 do begin
                    diff_sq= diff_sq+((sub_on[*,*,iband])[ind_common[icommon]]-fine2[i,j,iband])^2
                    diff_sq2= diff_sq2+((sub_on[*,*,iband])[ind_common[icommon]]-(sub_off[*,*,iband])[ind_common[icommon]])^2
                  endfor
                  rmsei[icommon]=(diff_sq/float(nb))^0.5+0.0001
                  rmse12[icommon]=(diff_sq2/float(nb))^0.5+0.0001
                  iw=ind_common[icommon] mod (a2-a1+1)
                  jw=ind_common[icommon]/(a2-a1+1)
                  disi[icommon]=((it[0]-iw[0])^2+(jt[0]-jw[0])^2)^0.5    ;the spatial distance
              endfor

               ind_similar=where(rmsei le similar_th,c_similar)

             if (c_similar lt min_similar and extent lt max_window ) then begin
                extent=extent+1
             endif else begin
               if (c_similar ge min_similar) then begin
                similar_rmse=rmsei[ind_similar]
                similar_rmse12=rmse12[ind_similar]                                         ;compute the weight W
                similar_dis=disi[ind_similar]
                C_D=similar_rmse*similar_dis
                weight=(1.0/C_D)/total(1.0/C_D)
                T_1=mean(similar_rmse)                                     ;compute the weight T
                T_2=mean(similar_rmse12)
                W_T1=T_2/(T_1+T_2)
                W_T2=T_1/(T_1+T_2)
                for iband=0,nb-1,1 do begin
                   similar_on=((sub_on[*,*,iband])[ind_common])[ind_similar]
                   similar_off=((sub_off[*,*,iband])[ind_common])[ind_similar]
                   predict_1=total(similar_off*weight)
                   predict_2=fine2[i,j,iband]+total((similar_off-similar_on)*weight)
                    if (predict_2 gt DN_min and predict_2 lt DN_max) then begin
                         fine0[i-location[0,isub],j-location[2,isub],iband]=W_T1*predict_1+W_T2*predict_2
                    endif else begin
                         fine0[i-location[0,isub],j-location[2,isub],iband]=predict_1
                    endelse
                endfor
                note=1
                mark[i-location[0,isub],j-location[2,isub]]=1+10*i_input
                gap[i,j]=2
                endif

                if (c_similar lt min_similar and extent eq max_window) then begin   ;IF reach the maximum window, use all the similar pixels
                   if( c_similar gt 0) then begin
                      similar_rmse=rmsei[ind_similar]
                      similar_rmse12=rmse12[ind_similar]
                      similar_dis=disi[ind_similar]
                      C_D=similar_rmse*similar_dis
                      weight=(1.0/C_D)/total(1.0/C_D)
                      T_1=mean(similar_rmse)
                      T_2=mean(similar_rmse12)
                      W_T1=T_2/(T_1+T_2)
                      W_T2=T_1/(T_1+T_2)
                      for iband=0,nb-1,1 do begin
                          similar_on=((sub_on[*,*,iband])[ind_common])[ind_similar]
                          similar_off=((sub_off[*,*,iband])[ind_common])[ind_similar]
                          predict_1=total(similar_off*weight)
                          predict_2=fine2[i,j,iband]+total((similar_off-similar_on)*weight)
                          if (predict_2 gt DN_min and predict_2 lt DN_max) then begin
                            fine0[i-location[0,isub],j-location[2,isub],iband]=W_T1*predict_1+W_T2*predict_2
                          endif else begin
                            fine0[i-location[0,isub],j-location[2,isub],iband]=predict_1
                          endelse
                      endfor
                      mark[i-location[0,isub],j-location[2,isub]]=2+10*i_input
                      note=1
                     endif
                     if( c_similar eq 0) then begin
                       for iband=0,nb-1,1 do begin
                         fine0[i-location[0,isub],j-location[2,isub],iband]=fine2[i,j,iband]+mean((sub_off[*,*,iband])[ind_common]-(sub_on[*,*,iband])[ind_common])
                         if (fine0[i-location[0,isub],j-location[2,isub],iband]) lt DN_min then begin
                            fine0[i-location[0,isub],j-location[2,isub],iband]=DN_min
                         endif
                         if (fine0[i-location[0,isub],j-location[2,isub],iband]) gt DN_max then begin
                            fine0[i-location[0,isub],j-location[2,isub],iband]=DN_max
                         endif
                       endfor
                         mark[i-location[0,isub],j-location[2,isub]]=3+10*i_input
                         note=1
                         gap[i,j]=2
                     endif
                 endif
              endelse
              endif else begin
                 if (extent lt max_window) then begin
                 extent=extent+1
                 endif else begin
                    if (c_common gt 0) then begin
                      for iband=0,nb-1,1 do begin
                        fine0[i-location[0,isub],j-location[2,isub],iband]=fine2[i,j,iband]+mean((sub_off[*,*,iband])[ind_common]-(sub_on[*,*,iband])[ind_common])
                         if (fine0[i-location[0,isub],j-location[2,isub],iband]) lt DN_min then begin
                            (fine0[i-location[0,isub],j-location[2,isub],iband])=DN_min
                         endif
                         if (fine0[i-location[0,isub],j-location[2,isub],iband]) gt DN_max then begin
                            (fine0[i-location[0,isub],j-location[2,isub],iband])=DN_max
                         endif
                      endfor
                        mark[i-location[0,isub],j-location[2,isub]]=3+10*i_input
                        note=1
                        gap[i,j]=2
                    endif else begin
                       extent=extent+1
                    endelse
                 endelse
              endelse
          endwhile
        endif
       endif
     endfor
    endfor

      ind_unfilled=where(gap eq 1, c_unfilled)
      if (c_unfilled eq 0) then begin
         note_finish=1
      endif else begin
         i_input=i_input+1
      endelse
       envi_file_mng, id=Fid2, /remove, /delete
   endwhile

   size_result=size(fine0)

    case Data_Type Of
        1:fine0 = Byte(fine0)    ;  BYTE  Byte
        2:fine0 = FIX(fine0)     ;  INT  Integer
        3:fine0 = LONG(fine0)    ;  LONG  Longword integer
        4:fine0 = FLOAT(fine0)   ;  FLOAT  Floating point
        5:fine0 = DOUBLE(fine0)  ;  DOUBLE  Double-precision floating
        6:fine0 = COMPLEX(fine0); complex, single-precision, floating-point
        9:fine0 = DCOMPLEX(fine0);complex, double-precision, floating-point
        12:fine0 = UINT(fine0)   ; unsigned integer vector or array
        13:fine0 = ULONG(fine0)   ;  unsigned longword integer vector or array
        14:fine0 = LONG64(fine0)   ;a 64-bit integer vector or array
        15:Ifine0 = ULONG64(fine0)   ;an unsigned 64-bit integer vector or array
    EndCase

       print,'finished ',isub+1,' block'

         Envi_Write_Envi_File,fine0,Out_Name = tempoutname1+strtrim(isub+1,1)
         Envi_Write_Envi_File,mark,Out_Name = tempoutname2+strtrim(isub+1,1)
         envi_file_mng, id=Fid1, /remove, /delete

endfor

;--------------------------------------------------------------------------------------
;mosiac all the filled patch

  mfid=intarr(n_ns*n_nl)
  mdims=intarr(5,n_ns*n_nl)
  mpos=intarr(nb,n_ns*n_nl)
  pos=indgen(nb)
  x0=intarr(n_ns*n_nl)
  y0=intarr(n_ns*n_nl)

  for isub=0,n_ns*n_nl-1,1 do begin
      envi_open_file, tempoutname1+strtrim(isub+1,1), r_fid= sub_fid
     if (sub_fid eq -1) then begin
       envi_batch_exit
       return
     endif
      envi_file_query,  sub_fid, ns=sub_ns, nl=sub_nl
      mfid[isub] = sub_fid
      mpos[*,isub] = indgen(nb)
      mdims[*,isub] = [-1,0, sub_ns-1,0, sub_nl-1]
      x0[isub]=ind_patch1[0,isub]
      y0[isub]=ind_patch1[2,isub]
  endfor

  xsize = orig_ns
  ysize = orig_nl
  pixel_size = [1.,1.]

  use_see_through = replicate(1L,n_ns*n_nl)
  see_through_val = replicate(0L,n_ns*n_nl)

;   out_name=Dialog_PickFile(Title = 'Enter the filename of the gap filled image')
    out_name=FileName1+'_filled_BNU'
    envi_doit, 'mosaic_doit', fid=mfid, pos=mpos, $
    dims=mdims, out_name=out_name, xsize=xsize, $
    ysize=ysize, x0=x0, y0=y0, georef=0,MAP_INFO=map_info, $
    out_dt=Data_Type, pixel_size=pixel_size, $
    background=0, see_through_val=see_through_val, $
    use_see_through=use_see_through

    for i=0,n_ns*n_nl-1,1 do begin
      envi_file_mng, id=mfid[i], /remove, /delete
    endfor

;--------------------------------------------------------------------------------------
;mosiac all the mask patch

  mfid=intarr(n_ns*n_nl)
  mdims=intarr(5,n_ns*n_nl)
  mpos=intarr(1,n_ns*n_nl)
  pos=indgen(1)
  x0=intarr(n_ns*n_nl)
  y0=intarr(n_ns*n_nl)

  for isub=0,n_ns*n_nl-1,1 do begin
      envi_open_file, tempoutname2+strtrim(isub+1,1), r_fid= sub_fid
     if (sub_fid eq -1) then begin
       envi_batch_exit
       return
     endif
      envi_file_query,  sub_fid, ns=sub_ns, nl=sub_nl
      mfid[isub] = sub_fid
      mpos[*,isub] = indgen(1)
      mdims[*,isub] = [-1,0, sub_ns-1,0, sub_nl-1]
      x0[isub]=ind_patch1[0,isub]
      y0[isub]=ind_patch1[2,isub]
  endfor

  xsize = orig_ns
  ysize = orig_nl
  pixel_size = [1.,1.]

  use_see_through = replicate(1L,n_ns*n_nl)
  see_through_val = replicate(0L,n_ns*n_nl)

;   out_name=Dialog_PickFile(Title = 'Enter the filename of the mask image')
    out_name=FileName1+'_mask'
    envi_doit, 'mosaic_doit', fid=mfid, pos=mpos, $
    dims=mdims, out_name=out_name, xsize=xsize, $
    ysize=ysize, x0=x0, y0=y0, georef=0,MAP_INFO=map_info, $
    out_dt=1, pixel_size=pixel_size, $
    background=0, see_through_val=see_through_val, $
    use_see_through=use_see_through

  for i=0,n_ns*n_nl-1,1 do begin
    envi_file_mng, id=mfid[i], /remove, /delete
  endfor

print, 'time used:', floor((systime(1)-t0)/3600), 'h',floor(((systime(1)-t0) mod 3600)/60),'m',(systime(1)-t0) mod 60,'s'


end