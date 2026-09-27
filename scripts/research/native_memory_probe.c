#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
int main(void) {
 const size_t n=256UL*1024*1024/sizeof(uint64_t);
 volatile uint64_t *p=malloc(n*sizeof(uint64_t));
 if(!p) return 2;
 unsigned long errors=0;
 for(unsigned round=0;round<8;round++) {
   uint64_t salt=UINT64_C(0x9e3779b97f4a7c15)*(round+1);
   for(size_t i=0;i<n;i++) p[i]=((uint64_t)i*UINT64_C(0xd6e8feb86659fd93))^salt;
   for(unsigned pass=0;pass<3;pass++)
     for(size_t i=0;i<n;i++) {
       uint64_t expected=((uint64_t)i*UINT64_C(0xd6e8feb86659fd93))^salt,actual=p[i];
       if(actual!=expected) {
         if(errors<12) printf("mismatch round=%u pass=%u index=%zu expected=%016" PRIx64 " actual=%016" PRIx64 " xor=%016" PRIx64 "\n",round,pass,i,expected,actual,expected^actual);
         errors++;
       }
     }
   printf("round=%u cumulative_mismatches=%lu\n",round,errors);fflush(stdout);
 }
 free((void*)p);return errors?1:0;
}
