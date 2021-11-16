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

import com.squareup.picasso.Picasso;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.recyclerview.widget.RecyclerView;

public class SpaceCardAdapter extends RecyclerView.Adapter<SpaceCardAdapter.Viewholder>{
    private Context context;
    private ArrayList<SpaceCard> SpaceCardArrayList;
    HashMap<Integer,Boolean> map = new HashMap<>();
    Integer [] num= {0,0,0,0};//소분류 개수 저장
    String []mcate_matching={"치킨","카페(도심)","유명관광지","햄버거/도너츠/샌드위치"};//인덱스에 따른 소분류 카테고리


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

        Picasso.with(context).load(card.getImage()).into(holder.spaceIV);
//        holder.spaceIV.setImageBitmap(card.getImage());
//        holder.spaceIV.setImageDrawable(card.getImage());
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
                    row+=card.getMlsfc()+",";//해당 장소 대분류 소분류 저장
                    row+=card.getMcate()+",";
                    //대분류와 소분류 갖고 오는 것으로 바꿀 것

                    switch (card.getMcate()){
                        case "치킨":
                            num[0]+=1;
                            break;

                        case "카페(도심)":
                            num[1]+=1;
                            break;
                        case "유명관광지":
                            num[2]+=1;
                            break;
                        case "햄버거/도너츠/샌드위치":
                            num[3]+=1;
                            break;
                    }
                    List<Integer> numlist = Arrays.asList(num);
                    int ind=numlist.indexOf(Collections.max(numlist));

                    //max를 받아와라

                    writeData2(row,mcate_matching[ind]);//파일 생성
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
//    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
//    public void writeData(String data){
//        String filePath="/data/data/com.example.refresh_selection/files";
//        String NEWLINE =System.lineSeparator();
//        File csv = new File(filePath+"/"+"data.csv");
//        BufferedWriter bw = null; // 출력 스트림 생성
//        if(csv.exists()){//있을때
//            Log.d("SpaceCardAdapter","파일 있네");
//            try {
//                bw = new BufferedWriter(new FileWriter(csv, true));
//                // csv파일의 기존 값에 이어쓰려면 위처럼 true를 지정하고, 기존 값을 덮어쓰려면 true를 삭제한다
//                bw.write(data+"\n");
//            } catch (FileNotFoundException e) {
//                e.printStackTrace();
//            } catch (IOException e) {
//                e.printStackTrace();
//            } finally {
//                try {
//                    if (bw != null) {
//                        bw.flush(); // 남아있는 데이터까지 보내 준다
//                        bw.close(); // 사용한 BufferedWriter를 닫아 준다
//                    }
//                } catch (IOException e) {
//                    e.printStackTrace();
//                }
//            }
//        }else{//없을 때
//            Log.d("SpaceCardAdapter","파일 생성");
//            try {
//                bw = new BufferedWriter(new FileWriter(csv));
//                // csv파일의 기존 값에 이어쓰려면 위처럼 true를 지정하고, 기존 값을 덮어쓰려면 true를 삭제한다
//                bw.write("mlsfc,mcate_nm");
//                bw.write(NEWLINE);
//                bw.write(data+"\n");
//                Log.d("SpaceCardAdapter","파일 씀");
//            } catch (FileNotFoundException e) {
//                e.printStackTrace();
//            } catch (IOException e) {
//                e.printStackTrace();
//            } finally {
//                try {
//                    if (bw != null) {
//                        bw.flush(); // 남아있는 데이터까지 보내 준다
//                        bw.close(); // 사용한 BufferedWriter를 닫아 준다
//                    }
//                } catch (IOException e) {
//                    e.printStackTrace();
//                }
//            }
//        }
//    }
    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    public void writeData2(String data,String fav_plc){
        String filePath="/data/data/com.example.refresh_selection/files";
        String NEWLINE =System.lineSeparator();
        File csv = new File(filePath+"/"+"input.csv");
        BufferedWriter bw = null; // 출력 스트림 생성
        List<String> g_a=readCSV();
        int len=g_a.size();
        String g_a_line=g_a.get(len-1);
        if(csv.exists()){//있을때
            Log.d("SpaceCardAdapter_2","파일 있네");
            try {
                bw = new BufferedWriter(new FileWriter(csv, true));
                // csv파일의 기존 값에 이어쓰려면 위처럼 true를 지정하고, 기존 값을 덮어쓰려면 true를 삭제한다
                bw.write(data+g_a_line+","+fav_plc+"\n");
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
            Log.d("SpaceCardAdapter_2","파일 생성");
            try {
                bw = new BufferedWriter(new FileWriter(csv));
                // csv파일의 기존 값에 이어쓰려면 위처럼 true를 지정하고, 기존 값을 덮어쓰려면 true를 삭제한다
                bw.write("mlsfc,mcate_nm,Sex,Age,Month,time,day,fav_plc");
                bw.write(NEWLINE);
                bw.write(data+g_a_line+","+fav_plc+"\n");
                Log.d("SpaceCardAdapter","파일 씀");
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
    public List<String> readCSV() {//List<List<String>>
//        List<List<String>> csvList = new ArrayList<List<String>>();
        List<String> csvList =new ArrayList<>();
        File csv = new File("/data/data/com.example.refresh_selection/files/gender_age.csv");
        BufferedReader br = null;
        String line = "";

        try {
            br = new BufferedReader(new FileReader(csv));
            while (null != (line = br.readLine())) { // readLine()은 파일에서 개행된 한 줄의 데이터를 읽어온다.
                List<String> aLine = new ArrayList<String>();
//                String[] lineArr = line.split(","); // 파일의 한 줄을 ,로 나누어 배열에 저장 후 리스트로 변환한다.
//                aLine = Arrays.asList(lineArr);
                csvList.add(line);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (br != null) {
                    br.close(); // 사용 후 BufferedReader를 닫아준다.
                }
            } catch(IOException e) {
                e.printStackTrace();
            }
        }
        return csvList;
    }
}
