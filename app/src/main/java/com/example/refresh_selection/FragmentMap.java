package com.example.refresh_selection;

import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import com.google.android.material.bottomsheet.BottomSheetBehavior;

import net.daum.mf.map.api.MapPOIItem;
import net.daum.mf.map.api.MapPoint;
import net.daum.mf.map.api.MapView;

import org.json.JSONException;
import org.json.JSONObject;

public class FragmentMap extends Fragment implements MapView.MapViewEventListener, ModalBottomSheet.BottomSheetListener {
    private String MarkerName;//마커 장소이름
    private double Latitude;//위도
    private double Longitude;//경도
    private int Tag;//마커 태그?
    private MapPOIItem testmarker;//마커객체?
    private int Radius = 3;//반경?
    private GpsTracker gpsTracker;//gpsTracker 객체
    //바텀시트
    private BottomSheetBehavior sheetBehavior;
    private ModalBottomSheet modal;
    LinearLayout layoutBottomSheet;

    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View v = inflater.inflate(R.layout.activity_map, container, false);
        View v2 = inflater.inflate(R.layout.bottom_sheet, container, false);
        MapView mapView = new MapView(getActivity());
        //마커 눌렀을때만? 바텀시트 띄우기
        showBottomSheetDialog();

        /**
         * bottom sheet state change listener
         * we are changing button text when sheet changed state
         * */
        layoutBottomSheet=v2.findViewById(R.id.standard_bottom_sheet);
        sheetBehavior = BottomSheetBehavior.from(layoutBottomSheet);

        sheetBehavior.setBottomSheetCallback(new BottomSheetBehavior.BottomSheetCallback() {
            @Override
            public void onStateChanged(@NonNull View bottomSheet, int newState) {
                switch (newState) {
                    case BottomSheetBehavior.STATE_HIDDEN:
                        break;
                    case BottomSheetBehavior.STATE_EXPANDED: {

                    }
                    break;
                    case BottomSheetBehavior.STATE_COLLAPSED: {

                    }
                    break;
                    case BottomSheetBehavior.STATE_DRAGGING:
                        break;
                    case BottomSheetBehavior.STATE_SETTLING:
                        break;
                }
            }

            @Override
            public void onSlide(@NonNull View bottomSheet, float slideOffset) {

            }
        });//바텀시트 행위


        //지도를 현재위치에 있는 장소로 띄어줌
        mapView.setMapViewEventListener(this);
        mapView.setCurrentLocationTrackingMode(MapView.CurrentLocationTrackingMode.TrackingModeOnWithHeading);
        //서버에 위도 경도요청하는 request보내기 반경을 요청하는 request 보내기
        //client가 반경을 줘야하나?
        mapView.getCurrentLocationTrackingMode();

        gpsTracker = new GpsTracker(getActivity());
        double lati = gpsTracker.getLatitude(); // 위도
         double longi = gpsTracker.getLongitude();// 경도
        // 필요시 String address = getCurrentAddress(latitude, longitude);
        Log.d("lati longi",lati+" "+longi+"");
        
        requestLatitudeLongtitude(lati,longi,Radius);//경도 위도 array를 받나?
        testmarker = setMarker(MarkerName, Latitude, Longitude, Tag);//서버에서 가져온 위도,경도,마커이름 //#태그번호는 임의로 지정?
        mapView.addPOIItem(testmarker);//마커 표시

        ViewGroup mapViewContainer = (ViewGroup) v.findViewById(R.id.map_view);//xml에 지도넣기
        mapViewContainer.addView(mapView);


        return v;
    }

    private void showBottomSheetDialog() {//하단 모달 시트 띄우는 메소드
        ModalBottomSheet bottomSheet=new ModalBottomSheet();
        bottomSheet.show(getChildFragmentManager(),"bottomSheet");
    }

    @Override
    public void onButtonClicked() {//ModalBottomSheet.BottomSheetListner

    }

    private void requestLatitudeLongtitude(double latitude, double longitude,int radius) {//서버에 장소 좌표 요청하는 메소드
        RequestQueue queue = Volley.newRequestQueue(getContext());
        String url = "";//서버url
        JSONObject testjson = new JSONObject();//서버에게 전달 할 값을 json으로 보냄
        try {
            testjson.put("longitude",longitude);
            testjson.put("latitude",latitude);
            testjson.put("radius", radius);
            testjson.put("requestLatitudeLongtitude", true);//서버에게 위도 경도를 요청한다.
        } catch (JSONException e) {
            e.printStackTrace();
        }
        JsonObjectRequest request = new JsonObjectRequest(Request.Method.GET, url, testjson, new Response.Listener() {
            @Override
            public void onResponse(Object response) {
                try {
                    JSONObject jsonResponse = new JSONObject(response.toString());
                    MarkerName = jsonResponse.getString("markerName");//마커 이름
                    Latitude = jsonResponse.getDouble("latitude");//마커 위도
                    Longitude = jsonResponse.getDouble("longitude");//마커 경도
                    Tag = jsonResponse.getInt("tag");//마커 구분 번호?
                } catch (JSONException e) {
                    e.printStackTrace();
                }
            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Log.d("responseError", "위도, 경도 가져오기 실패");
            }
        });
        request.setRetryPolicy(new DefaultRetryPolicy(DefaultRetryPolicy.DEFAULT_TIMEOUT_MS, DefaultRetryPolicy.DEFAULT_MAX_RETRIES, DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
        queue.add(request);
    }

    private MapPOIItem setMarker(String markerName, double latitude, double longitude, int tag) {
        //마커객체 생성 메소드
        MapPOIItem marker = new MapPOIItem();

        //맵 포인트 위도경도 설정 나중에 서버에서 제공하는 위도 ,경도로 설정
        MapPoint mapPoint = MapPoint.mapPointWithGeoCoord(latitude, longitude);//36.369015338611675, 127.34685766817087 충남대학교 좌표
        marker.setItemName(markerName);//마커이름
        marker.setTag(tag);//마커 태그?
        marker.setMapPoint(mapPoint);//마커를 위치에 표시
        marker.setMarkerType(MapPOIItem.MarkerType.BluePin); // 기본으로 제공하는 BluePin 마커 모양.
        marker.setSelectedMarkerType(MapPOIItem.MarkerType.RedPin); // 마커를 클릭했을때, 기본으로 제공하는 RedPin 마커 모양.
        return marker;
    }

    //MapView.MapViewEventListener 구현
    @Override
    public void onMapViewInitialized(MapView mapView) {

    }

    @Override
    public void onMapViewCenterPointMoved(MapView mapView, MapPoint mapCenterPoint) {

    }

    @Override
    public void onMapViewZoomLevelChanged(MapView mapView, int i) {

    }

    @Override
    public void onMapViewSingleTapped(MapView mapView, MapPoint mapPoint) {

    }

    @Override
    public void onMapViewDoubleTapped(MapView mapView, MapPoint mapPoint) {

    }

    @Override
    public void onMapViewLongPressed(MapView mapView, MapPoint mapPoint) {

    }

    @Override
    public void onMapViewDragStarted(MapView mapView, MapPoint mapPoint) {

    }

    @Override
    public void onMapViewDragEnded(MapView mapView, MapPoint mapPoint) {

    }

    @Override
    public void onMapViewMoveFinished(MapView mapView, MapPoint mapPoint) {

    }


}