for /l %%x in (1, 1, 55) do (
   echo %%x
   ren "img (%%x).JPG" image%%x.jpg
)