package com.example.refresh_selection;

import android.content.Context;
import android.os.Build;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
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
import java.util.List;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.recyclerview.widget.RecyclerView;

public class BottomCardAdapter extends RecyclerView.Adapter<BottomCardAdapter.Viewholder> {
    private Context context;
    private ArrayList<BottomCard> BottomCardArrayList;
    String []mcate_matching={"치킨","카페(도심)","유명관광지","햄버거/도너츠/샌드위치"};//소분류 개수 저장
    Integer [] num= {0,0,0,0};
    List<String> mcate_m;
    public void InitArray(){
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(
                    new FileReader("C:\\Users\\SAMSUNG\\Desktop\\hyorin_refresh\\refresh_selection_frontend\\app\\src\\main\\assets\\mcate_nm_vocab.txt")
            );
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        String str = null;
        mcate_m=new ArrayList<>();
        while (true) {
            try {
                if (!((str = reader.readLine()) != null)) break;
            } catch (IOException e) {
                e.printStackTrace();
            }
            mcate_m.add(str);//{"떡집 0","피자 1","치킨 2"}이런식
        }
        try {
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
    //생성자
    public BottomCardAdapter(Context context, ArrayList<BottomCard> BottomCardArrayList){
        this.context=context;
        this.BottomCardArrayList=BottomCardArrayList;
    }

    @NonNull
    @Override
    public BottomCardAdapter.Viewholder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.bottom_cardview,parent,false);
        return new Viewholder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull BottomCardAdapter.Viewholder holder, int position) {
        BottomCard card = BottomCardArrayList.get(position);
        Picasso.with(context).load(card.getImg()).into(holder.btmIV);
        holder.btmNameTV.setText(card.getBottom_space_name());
        holder.btmDesTV.setText(card.getBottom_space_description());
        holder.btmDis.setText(card.getDistance());
        holder.btmPrice.setText(card.getPrice());

        holder.like_bt.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.KITKAT)
            @Override
            public void onClick(View v) {
                if(holder.check_like==false||holder.like==true){
                    if(holder.like==false){
                        holder.like_bt.setImageResource(R.drawable.like_bt);//활성화
                        holder.setLike(true);
                        String row="";
                        row+=card.getMlsfc()+",";//해당 장소 대분류 소분류 저장
                        row+=card.getMcate()+",";

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
                        writeData2(row,mcate_matching[ind]);
                        holder.check_like=true;//like dislike중 like 선택
                    }else{
                        holder.like_bt.setImageResource(R.drawable.unactivated_like);//비활성화
                        holder.setLike(false);
                        holder.check_like=false;
                    }

                }


            }
        });

        holder.dislike_bt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(holder.check_like==false||holder.dislike==true){//선택한거 없을때나
                    if(holder.dislike==false){
                        holder.dislike_bt.setImageResource(R.drawable.dislike_bt);//활성화
                        holder.setDislike(true);
                        holder.check_like=true;//like dislike중 dislike 선택
                    }else{
                        holder.dislike_bt.setImageResource(R.drawable.unactivated_dislike);//비활성화
                        holder.setDislike(false);
                        holder.check_like=false;
                    }

                }


            }
        });
    }

    @Override
    public int getItemCount() {
        return BottomCardArrayList.size();
    }

    public class Viewholder extends RecyclerView.ViewHolder {
        private ImageView btmIV;
        private TextView btmNameTV,btmDesTV,btmDis,btmPrice;
        private ImageView like_bt, dislike_bt;
        private Boolean check_like=false;
        //like와 dislike 둘다 선택하는거 막음 둘중 하나만 선택 가능
        private Boolean like=false, dislike=false;//like_bt dislike_bt 각각 선택되었는지 여부
        // 2번째 클릭할때 deactive이미지로 바뀜

        public Viewholder(@NonNull View itemView) {
            super(itemView);
            btmIV= itemView.findViewById(R.id.btmIV);
            btmNameTV=itemView.findViewById(R.id.bottom_space_name);
            btmDesTV=itemView.findViewById(R.id.bottom_space_description);
            btmDis=itemView.findViewById(R.id.bottom_space_distance);
            btmPrice=itemView.findViewById(R.id.bottom_space_price);
            like_bt=itemView.findViewById(R.id.like_bt);
            dislike_bt=itemView.findViewById(R.id.dislike_bt);


        }

        public Boolean getLike() {
            return like;
        }

        public void setLike(Boolean like) {
            this.like = like;
        }

        public Boolean getDislike() {
            return dislike;
        }

        public void setDislike(Boolean dislike) {
            this.dislike = dislike;
        }

        public Boolean getCheck_like() {
            return check_like;
        }

        public void setCheck_like(Boolean check_like) {
            this.check_like = check_like;
        }
    }

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
