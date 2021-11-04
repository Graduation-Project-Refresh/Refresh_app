package com.example.refresh_selection;

import android.content.Context;
import android.os.Build;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.recyclerview.widget.RecyclerView;

public class SpaceCardAdapter extends RecyclerView.Adapter<SpaceCardAdapter.Viewholder>{
    private Context context;
    private ArrayList<SpaceCard> SpaceCardArrayList;
    HashMap<Integer,Boolean> map = new HashMap<>();

    //생성자
    public SpaceCardAdapter(Context context, ArrayList<SpaceCard> SpaceCardArrayList){
        this.context=context;
        this.SpaceCardArrayList=SpaceCardArrayList;
    }
    @NonNull
    @Override
    public SpaceCardAdapter.Viewholder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.cardview,parent,false);
        return new Viewholder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull SpaceCardAdapter.Viewholder holder, int position) {
        SpaceCard card =SpaceCardArrayList.get(position);
        holder.spaceIV.setImageDrawable(card.getImage());
//        holder.spaceIV.setImageResource(card.getImage());
        holder.spaceNameTV.setText(card.getSpace_name());
        Log.d("space_name",card.getSpace_name());
        holder.spaceDescriptionTV.setText(card.getDescription());
        holder.spaceDescription2TV.setText(card.getDescription2());
        holder.heart_bt.setOnClickListener(new View.OnClickListener(){

            @RequiresApi(api = Build.VERSION_CODES.KITKAT)
            @Override
            public void onClick(View v) {
                if(holder.check_heart==false){
                    holder.heart_bt.setImageResource(R.drawable.heart_bt_active);
                    holder.check_heart=true;//활성화
                    map.put(position,true);//모델이 학습해야할 카드뷰의 위치 저장
                    String row="";
                    row+=card.getSpace_name()+",";//해당 장소 대분류 소분류 저장
                    row+=card.getDescription()+",";
                    //대분류와 소분류 갖고 오는 것으로 바꿀 것

                    Date dt = new Date();
                    String month=(dt.getMonth()+1)+"";//month+1 한 값이 월
                    String hour=dt.getHours()+"";
                    SimpleDateFormat sdf = new SimpleDateFormat("EE");//요일
                    String day= sdf.format(dt).toString()+"요일";//요일
                    row+=month+","+hour+","+day;
//                    SimpleDateFormat full_sdf = new SimpleDateFormat("yyyy-MM-dd, hh:mm:ss a");
                    Log.d("DATE",row);
                    writeData(row);//파일 생성
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                        Log.d("dataDirPath", "Internal private data dir: " + context.getDataDir().getAbsolutePath());
                    }
                    Log.d("fileDirPath", "Internal private file dir: " + context.getFilesDir().getAbsolutePath());
                    Log.d("CacheDirPath", "Internal private cache dir: " + context.getCacheDir().getAbsolutePath());
                    String filePath = context.getFilesDir().getAbsolutePath();
//                    ExcelWriter ew=new ExcelWriter();
//                    ew.writeCSV(row,"data");

                    //해당 장소 좋아요를 클릭한 날짜를 Excel에 저장

                }else{
                    holder.heart_bt.setImageResource(R.drawable.heart_bt);
                    holder.check_heart=false;//비활성화
                    map.put(position,false);// 모델이 학습해야할 카드뷰의 위치 삭제
                }

            }
        });


    }
    public void createInfoData(){

    }

    @Override
    public int getItemCount() {
        return SpaceCardArrayList.size();
    }
    // View holder class for initializing of
    // your views such as TextView and Imageview.
    public class Viewholder extends RecyclerView.ViewHolder {
        private ImageView spaceIV;
        private TextView spaceNameTV, spaceDescriptionTV,spaceDescription2TV;
        private ImageButton heart_bt,cal_bt;
        private Boolean check_heart=false;
        public Viewholder(@NonNull View itemView) {
            super(itemView);
            spaceIV = itemView.findViewById(R.id.space_image);
            spaceNameTV = itemView.findViewById(R.id.space_name);
            spaceDescriptionTV = itemView.findViewById(R.id.space_description);
            spaceDescription2TV = itemView.findViewById(R.id.space_description2);
            heart_bt=itemView.findViewById(R.id.heart_bt);
            cal_bt= itemView.findViewById(R.id.cal_bt);
        }
    }
    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    public void writeData(String data){
        String filePath="/data/data/com.example.refresh_selection/files";
        String NEWLINE =System.lineSeparator();
        File csv = new File(filePath+"/"+"data.csv");
        BufferedWriter bw = null; // 출력 스트림 생성
        if(csv.exists()){//있을때
            try {
                bw = new BufferedWriter(new FileWriter(csv, true));
                // csv파일의 기존 값에 이어쓰려면 위처럼 true를 지정하고, 기존 값을 덮어쓰려면 true를 삭제한다
                bw.write(data+"\n");
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                try {
                    if (bw != null) {
                        bw.flush(); // 남아있는 데이터까지 보내 준다
                        bw.close(); // 사용한 BufferedWriter를 닫아 준다
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }else{//없을 때
            try {
                bw = new BufferedWriter(new FileWriter(csv));
                // csv파일의 기존 값에 이어쓰려면 위처럼 true를 지정하고, 기존 값을 덮어쓰려면 true를 삭제한다
                bw.write("mlsfc,mcate_nm,Month,time,day");
                bw.write(NEWLINE);
                bw.write(data+"\n");
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                try {
                    if (bw != null) {
                        bw.flush(); // 남아있는 데이터까지 보내 준다
                        bw.close(); // 사용한 BufferedWriter를 닫아 준다
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

    }
}
