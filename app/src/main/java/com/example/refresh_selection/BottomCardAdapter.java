package com.example.refresh_selection;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.ArrayList;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

public class BottomCardAdapter extends RecyclerView.Adapter<BottomCardAdapter.Viewholder> {
    private Context context;
    private ArrayList<BottomCard> BottomCardArrayList;

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
        holder.btmIV.setImageResource(card.getImg());
        holder.btmNameTV.setText(card.getBottom_space_name());
        holder.btmDesTV.setText(card.getBottom_space_description());
        holder.btmDis.setText(card.getDistance());
        holder.btmPrice.setText(card.getPrice());

        holder.like_bt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(holder.check_like==false||holder.like==true){
                    if(holder.like==false){
                        holder.like_bt.setImageResource(R.drawable.like_bt);//활성화
                        holder.setLike(true);
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
}
