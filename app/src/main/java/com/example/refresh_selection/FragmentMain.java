package com.example.refresh_selection;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;

public class FragmentMain extends Fragment {
    SpaceCardAdapter spaceCardAdapter;
    private RecyclerView spaceRV;
    private ArrayList<SpaceCard> spaceCardArrayList;//카드 정보 저장
//    private ViewGroup mainview;
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        ViewGroup mainview =(ViewGroup)inflater.inflate(R.layout.activity_main3, container, false);
        makeSpaceCard(mainview);

        return mainview;
    }

    public void makeSpaceCard(ViewGroup mainview) {
        spaceRV = (RecyclerView) mainview.findViewById(R.id.idRVSpace);//main_activity_3
        spaceRV.setHasFixedSize(true);
        spaceCardArrayList= new ArrayList<>();
        spaceCardArrayList.add(new SpaceCard("대청호1","호수가 아름답다.","여기는 큰 호수이다.",R.drawable.test_img));
        spaceCardArrayList.add(new SpaceCard("대청호2","호수가 아름답다.","여기는 큰 호수이다.",R.drawable.test_img));
        spaceCardArrayList.add(new SpaceCard("대청호3","호수가 아름답다.","여기는 큰 호수이다.",R.drawable.test_img));

        spaceCardAdapter =new SpaceCardAdapter(mainview.getContext(),spaceCardArrayList);
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(mainview.getContext(), LinearLayoutManager.VERTICAL, false);
        spaceRV.setLayoutManager(linearLayoutManager);
        spaceRV.setAdapter(spaceCardAdapter);
    }

//    @Override
//    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
//        super.onActivityCreated(savedInstanceState);
//        spaceRV.setAdapter(spaceCardAdapter);
//    }
//    @Override
//    public void onAttach(Activity activity) {
//        super.onAttach(activity);
//        spaceCardArrayList= new ArrayList<>();
//        spaceCardArrayList.add(new SpaceCard("대청호1","호수가 아름답다.","여기는 큰 호수이다.",R.drawable.test_img));
//        spaceCardArrayList.add(new SpaceCard("대청호2","호수가 아름답다.","여기는 큰 호수이다.",R.drawable.test_img));
//        spaceCardArrayList.add(new SpaceCard("대청호3","호수가 아름답다.","여기는 큰 호수이다.",R.drawable.test_img));
//
//        spaceCardAdapter =new SpaceCardAdapter(activity,spaceCardArrayList);
//        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(activity, LinearLayoutManager.VERTICAL, false);
//        spaceRV.setLayoutManager(linearLayoutManager);
//        spaceRV.setAdapter(spaceCardAdapter);
//    }
//    @Override
//    public View getView(){
//        return mainview;
//    }

}
